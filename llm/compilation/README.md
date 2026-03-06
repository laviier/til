# LLM Compilation — Hands-On Learning

Breaking down the full GPU compilation stack, one script at a time.
Each script covers one layer. Write them yourself from scratch.

## Layers to cover

0. `00_eager_baseline.py` — Eager PyTorch, measuring per-op overhead
1. `01_dynamo_graph_capture.py` — TorchDynamo: graph export, guards, graph breaks
2. `02_fx_graph.py` — FX Graph: nodes, transformations, fusion passes
3. `03_inductor.py` — Inductor: compiling, inspecting generated code, benchmarking
4. `04_triton_kernels.py` — Triton: write your first GPU kernels
5. `05_ptx_inspection.py` — PTX/SASS: extracting and reading GPU assembly
6. `06_gpu_hardware.py` — Measuring bandwidth, TFLOPS, launch overhead
7. `07_kernel_zoo.py` — cuBLAS vs CUTLASS vs Triton benchmarks
8. `08_cuda_graphs.py` — CUDA Graph capture and replay
9. `09_full_pipeline.py` — Everything together, end-to-end benchmark
10. `10_compile_vs_export.py` — torch.compile (JIT) vs torch.export (AOT): differences, graph breaks, serialization

## Deep Dive
### TorchDynamo

https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html

**Graph Breaks**: When Dynamo encounters something it can't trace (like `print()`, complex Python control flow, or unsupported operations), it "breaks" the graph. The code before the break becomes one graph, the code after becomes another. Between them, Python runs normally. Fewer graph breaks = better optimization.

**Guards**: Dynamo records assumptions about the input (e.g., "x has shape [32, 4096]" or "x.dtype is float16"). If these assumptions are violated on a future call, Dynamo recompiles. This is called a "guard failure."

**Symbolic Shapes**: Instead of recording concrete shapes like `[32, 4096]`, Dynamo can record symbolic shapes like `[s0, 4096]` where `s0` is a symbol that can take different values. This avoids recompilation when batch size changes.

**FX Graph**: An FX Graph is PyTorch's **intermediate representation (IR)** — a data structure that represents your computation as a directed acyclic graph (DAG) of operations. It's like a recipe that says "do this, then this, then this" but without actually doing anything.

```
┌─────────────────────────────────────────────────────┐
│                    Python Code                      │
│  def forward(self, x):                              │
│      y = self.linear(x)     # torch.mm + bias add   │
│      z = torch.relu(y)      # torch.relu            │
│      return z                                       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              TorchDynamo (Bytecode Analysis)        │
│                                                     │
│  1. Intercepts Python frame evaluation              │
│  2. Symbolically executes each bytecode instruction │
│  3. For tensor ops: records them in an FX graph     │
│  4. For Python control flow: creates "graph breaks" │
│  5. For data-dependent branches: may specialize     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              FX Graph (Captured Computation)        │
│                                                     │
│  placeholder: x (shape=[batch, hidden])             │
│  call_function: mm(x, weight.T)                     │
│  call_function: add(result, bias)                   │
│  call_function: relu(result)                        │
│  output: result                                     │
└─────────────────────────────────────────────────────┘
```

### Graph Splitting (Piecewise Compilation)

https://blog.vllm.ai/2025/08/20/torch-compile.html

This is a critical vLLM innovation. The full graph is **split** at attention operations:

```
┌──────────────────────────────────────────────────────────────┐
│                    Full Computation Graph                    │
│                                                              │
│  [RMSNorm → QKV Proj] → [ATTENTION] → [OutProj → Residual    │
│   → RMSNorm → Gate/Up] → [ATTENTION] → [Down → Residual]     │
│                                                              │
│  Split into pieces:                                          │
│                                                              │
│  Piece 0: [RMSNorm → QKV Proj]        ← Compiled by Inductor │
│  Piece 1: [ATTENTION]                 ← Runs as custom op    │
│  Piece 2: [OutProj → Residual → ... ] ← Compiled by Inductor │
│  Piece 3: [ATTENTION]                 ← Runs as custom op    │
│  Piece 4: [Down → Residual]           ← Compiled by Inductor │
└──────────────────────────────────────────────────────────────┘
```

Why split at attention?
- Attention has complex, variable-length behavior (different sequence lengths, KV cache interactions)
- Attention already has highly optimized hand-written kernels (FlashAttention, etc.)
- The pieces between attention ops are "simple" element-wise or matrix operations that Inductor can fuse beautifully

Input / Output:
| | Description |
|---|---|
| **Input** | `torch.fx.GraphModule` from Dynamo |
| **Output** | List of `SplitItem` objects, each containing a sub-graph to be compiled independently |

### Inductor

Inductor is PyTorch's **default compiler backend**. It takes an FX graph and produces actual GPU code (Triton kernels or C++/CUDA code). Think of it as the "code generator" — it decides how to implement each operation efficiently.

```
┌─────────────────────────────────────────────────────────────┐
│                    FX Graph (from Dynamo)                   │
│  mm → add → relu → mm → silu → mul                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 1: Graph Lowering                         │
│                                                             │
│  Convert high-level ops to lower-level primitives:          │
│  - nn.Linear → mm + add                                     │
│  - SiLU → sigmoid * x                                      │
│  - RMSNorm → variance → rsqrt → mul                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 2: Fusion Analysis                        │
│                                                             │
│  Identify operations that can be combined:                  │
│  - Element-wise chains: norm → mul → add → relu             │
│  - Reduction + element-wise: softmax = exp → sum → div      │
│  - Memory-bound ops benefit most from fusion                │
│                                                             │
│  Example fusion:                                            │
│  BEFORE: read(x) → relu(x) → write(y) → read(y) → mul(y,2)  │
│  AFTER:  read(x) → relu(x) → mul(result,2) → write(y)       │
│        (saved 1 read + 1 write = huge memory bandwidth win) │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 3: Code Generation                        │
│                                                             │
│  For each fused group, generate a Triton kernel:            │
│                                                             │
│  @triton.jit                                                │
│  def fused_relu_mul(x_ptr, out_ptr, n_elements, BLOCK: ...):│
│      pid = tl.program_id(0)                                 │
│      offsets = pid * BLOCK + tl.arange(0, BLOCK)            │
│      x = tl.load(x_ptr + offsets, mask=offsets < n_elements)│
│      result = tl.maximum(x, 0) * 2  # relu + mul fused!     │
│      tl.store(out_ptr + offsets, result, ...)               │
│                                                             │
│  For matrix multiplications:                                │
│  → May use Triton matmul template OR dispatch to cuBLAS     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 4: Auto-Tuning (Optional)                 │
│                                                             │
│  When shapes are known (compile_sizes=[1,2,4,8]):           │
│  Try different kernel configurations:                       │
│  - BLOCK_M=16, BLOCK_N=32, num_warps=2  → 0.013ms ✓ best    │
│  - BLOCK_M=16, BLOCK_N=64, num_warps=4  → 0.013ms           │
│  - BLOCK_M=16, BLOCK_N=128, num_warps=4 → 0.015ms           │
│  - cuBLAS default                        → 0.016ms          │
└─────────────────────────────────────────────────────────────┘
```

The output is a Python file containing Triton kernel definitions and a "call" function that orchestrates them. vLLM adds its own optimization passes on top of Inductor's defaults:

```python
# From vllm/config/compilation.py - PassConfig
class PassConfig:
    fuse_norm_quant: bool    # Fuse RMSNorm + FP8 quantization
    fuse_act_quant: bool     # Fuse SiLU*Mul + FP8 quantization  
    fuse_attn_quant: bool    # Fuse Attention output + FP8 quantization
    enable_sp: bool          # Sequence parallelism across tensor-parallel ranks
    fuse_gemm_comms: bool    # Overlap GEMM computation with all-reduce communication
    fuse_allreduce_rms: bool # Fuse all-reduce + RMSNorm (FlashInfer)
```
These passes operate on the FX graph before Inductor generates code, inserting custom fused operations.

Input / Output:
| | Description |
|---|---|
| **Input** | FX sub-graph (one piece from the split), example inputs with shapes |
| **Output** | Compiled Python module containing Triton kernels + orchestration code, cached to disk |

### Triton (High-Level GPU Kernel Language)

Triton is a **programming language and compiler for writing GPU kernels** created by OpenAI. It sits between Python (too high-level for GPU programming) and CUDA C++ (too low-level and error-prone). Triton lets you write GPU code that looks almost like NumPy but compiles to highly optimized GPU machine code.

Writing CUDA kernels requires managing:
- Thread indexing (which thread does what)
- Shared memory (fast on-chip memory)
- Memory coalescing (accessing memory in patterns the GPU likes)
- Warp synchronization (coordinating groups of 32 threads)
- Register pressure (not using too many registers per thread)

Triton abstracts all of this. You think in terms of **blocks of data** instead of individual threads.

Here's a real kernel from vLLM that merges attention states (simplified with annotations):

```python
@triton.jit  # This decorator tells Triton to compile this function for GPU
def merge_attn_states_kernel(
    output,          # Pointer to output tensor in GPU memory
    prefix_output,   # Pointer to prefix attention output
    prefix_lse,      # Pointer to prefix log-sum-exp values
    suffix_output,   # Pointer to suffix attention output  
    suffix_lse,      # Pointer to suffix log-sum-exp values
    HEAD_SIZE: tl.constexpr,        # Known at compile time
    PADDED_HEAD_SIZE: tl.constexpr, # Known at compile time
):
    # ─── STEP 1: Figure out which piece of data this "program" handles ───
    # Each "program instance" handles one (token, head) pair
    # tl.program_id(0) is like "which block am I?" in CUDA
    token_idx = tl.program_id(0)   # Which token
    head_idx = tl.program_id(1)    # Which attention head
    
    # ─── STEP 2: Load data from GPU global memory ───
    # Load the log-sum-exp values (scalars)
    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)
    
    # ─── STEP 3: Compute (this happens in registers, super fast) ───
    max_lse = tl.maximum(p_lse, s_lse)
    p_se = tl.exp(p_lse - max_lse)
    s_se = tl.exp(s_lse - max_lse)
    out_se = p_se + s_se
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    
    # Load vectors (a whole HEAD_SIZE chunk at once)
    head_arange = tl.arange(0, PADDED_HEAD_SIZE)  # [0, 1, 2, ..., 127]
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(prefix_output + ..., mask=head_mask)
    s_out = tl.load(suffix_output + ..., mask=head_mask)
    
    # Weighted combination
    out = p_out * p_scale + s_out * s_scale
    
    # ─── STEP 4: Write result back to GPU global memory ───
    tl.store(output + ..., out, mask=head_mask)
```

How Triton Launches Work

```python
# From Python, you launch the kernel with a "grid" specification:
merge_attn_states_kernel[(num_tokens, num_query_heads)](
    output, prefix_output, prefix_lse, suffix_output, suffix_lse,
    HEAD_SIZE=128, PADDED_HEAD_SIZE=128,
)
# The grid (num_tokens, num_query_heads) means:
# - Launch num_tokens × num_query_heads "program instances"
# - Each instance runs the kernel function independently
# - The GPU schedules these across its Streaming Multiprocessors (SMs)
```

Triton Compilation Pipeline

```
┌──────────────────────────────────────────────────────┐
│           Triton Python Code (@triton.jit)           │
│  def my_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):   │
│      pid = tl.program_id(0)                          │
│      x = tl.load(x_ptr + pid * BLOCK + arange)       │
│      tl.store(y_ptr + ..., tl.exp(x))                │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│           Triton IR (MLIR-based)                     │
│  Triton's own intermediate representation            │
│  - Block-level operations                            │
│  - Memory access patterns                            │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│           LLVM IR                                    │
│  Standard compiler intermediate representation       │
│  - Loop unrolling, vectorization                     │
│  - Register allocation                               │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│           PTX (Parallel Thread Execution)            │
│  NVIDIA's virtual GPU assembly language              │
│  - Virtual registers                                 │
│  - Memory operations (ld.global, st.shared)          │
│  - Arithmetic (fma.f32, mul.f16)                     │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│           SASS (Shader Assembly)                     │
│  Actual GPU machine code (binary)                    │
│  - Hardware-specific instructions                    │
│  - Scheduled for specific GPU architecture           │
└──────────────────────────────────────────────────────┘
```

Input / Output:
| | Description |
|---|---|
| **Input** | Triton Python function decorated with `@triton.jit`, grid dimensions, kernel arguments (pointers to GPU memory + constants) |
| **Output** | Compiled GPU binary (cubin) loaded into GPU memory, ready to execute |

### PTX / SASS (GPU Assembly)

#### PTX: The Virtual Assembly

PTX (Parallel Thread Execution) is NVIDIA's **virtual instruction set architecture (ISA)**. It's like x86 assembly but for GPUs. It's "virtual" because it's not the final machine code — the GPU driver compiles PTX to actual hardware instructions (SASS) at load time or ahead of time.

```asm
; PTX example: Fused multiply-add on float32
; result = a * b + c
.reg .f32 %a, %b, %c, %result;

ld.global.f32 %a, [%rd1];        // Load 'a' from global memory
ld.global.f32 %b, [%rd2];        // Load 'b' from global memory  
ld.global.f32 %c, [%rd3];        // Load 'c' from global memory
fma.rn.f32 %result, %a, %b, %c;  // result = a*b + c (one instruction!)
st.global.f32 [%rd4], %result;    // Store result to global memory
```

Key PTX concepts:
- **Registers** (`.reg`): Super-fast per-thread storage
- **Memory spaces**: `global` (slow, large), `shared` (fast, small, shared within a block), `local` (per-thread)
- **Fused operations**: `fma` (fused multiply-add) does 2 operations in 1 instruction

#### SASS: The Real Machine Code

SASS is the actual binary instruction set that runs on a specific GPU architecture (e.g., SM_90 for Hopper, SM_80 for Ampere). You rarely see SASS directly, but it's what the GPU actually executes.

```
; SASS example (Hopper SM_90) — you'd see this in nsight profiler
FFMA R4, R0, R2, R4;     // Fused float multiply-add
LDG.E R0, [R6];          // Load from global memory
STG.E [R8], R4;          // Store to global memory
HMMA.16816.F32 ...;      // Tensor Core matrix multiply-accumulate
```

#### The Compilation Chain

```
Triton/CUDA C++ → PTX (virtual) → SASS (real hardware instructions)
                     │                    │
                     │                    └── Done by NVIDIA driver (JIT)
                     │                        or nvcc (ahead of time)
                     └── Done by Triton compiler or nvcc
```

Input / Output:
| | Description |
|---|---|
| **Input** | Triton IR or CUDA C++ source code |
| **Output** | `.cubin` file (GPU binary) containing SASS instructions, loaded into GPU memory |


GPU Hardware — SMs, Warps, and Threads

### GPU Architecture Overview

A modern NVIDIA GPU (e.g., H100) is organized like this:

```
┌────────────────────────────────────────────────────────────────┐
│                         GPU (e.g., H100)                       │
│                                                                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐           │
│  │  SM 0   │ │  SM 1   │ │  SM 2   │ ... │ SM 131  │  (132 SMs)│
│  │         │ │         │ │         │     │         │           │
│  │ 64 CUDA │ │ 64 CUDA │ │ 64 CUDA │     │ 64 CUDA │           │
│  │ Cores   │ │ Cores   │ │ Cores   │     │ Cores   │           │
│  │         │ │         │ │         │     │         │           │
│  │ 4 Tensor│ │ 4 Tensor│ │ 4 Tensor│     │ 4 Tensor│           │
│  │ Cores   │ │ Cores   │ │ Cores   │     │ Cores   │           │
│  │         │ │         │ │         │     │         │           │
│  │ 256KB   │ │ 256KB   │ │ 256KB   │     │ 256KB   │           │
│  │ Shared  │ │ Shared  │ │ Shared  │     │ Shared  │           │
│  │ Memory  │ │ Memory  │ │ Memory  │     │ Memory  │           │
│  │ /L1     │ │ /L1     │ │ /L1     │     │ /L1     │           │
│  └─────────┘ └─────────┘ └─────────┘     └─────────┘           │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    L2 Cache (50 MB)                       │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              HBM3 Global Memory (80 GB, 3.35 TB/s)        │ │
│  └───────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

An SM is the **fundamental compute unit** of a GPU. Think of it as a mini-processor that can run many threads simultaneously. Each SM contains:

| Component | What It Does | H100 Specs |
|---|---|---|
| **CUDA Cores** | General-purpose floating-point and integer arithmetic (add, multiply, etc.) | 128 FP32 cores per SM |
| **Tensor Cores** | Specialized matrix multiply-accumulate units (for AI workloads) | 4 per SM (4th gen) |
| **Shared Memory / L1 Cache** | Fast on-chip memory shared by all threads in a block | 256 KB configurable |
| **Register File** | Ultra-fast per-thread storage | 256 KB (65536 × 32-bit registers) |
| **Warp Schedulers** | Hardware units that schedule groups of 32 threads | 4 per SM |
| **Special Function Units (SFUs)** | For transcendental math (sin, cos, exp, rsqrt) | 16 per SM |


#### Thread Hierarchy: How Work Gets Distributed

```
┌────────────────────────────────────────────────────────────┐
│                        GRID                                │
│  (All the work for one kernel launch)                      │
│                                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Block 0  │ │ Block 1  │ │ Block 2  │ │ Block N  │       │
│  │          │ │          │ │          │ │          │       │
│  │ ┌──────┐ │ │ ┌──────┐ │ │          │ │          │       │
│  │ │Warp 0│ │ │ │Warp 0│ │ │  ...     │ │  ...     │       │
│  │ │32 thr│ │ │ │32 thr│ │ │          │ │          │       │
│  │ ├──────┤ │ │ ├──────┤ │ │          │ │          │       │
│  │ │Warp 1│ │ │ │Warp 1│ │ │          │ │          │       │
│  │ │32 thr│ │ │ │32 thr│ │ │          │ │          │       │
│  │ ├──────┤ │ │ ├──────┤ │ │          │ │          │       │
│  │ │ ...  │ │ │ │ ...  │ │ │          │ │          │       │
│  │ └──────┘ │ │ └──────┘ │ │          │ │          │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                            │
│  Blocks are assigned to SMs by the GPU hardware scheduler  │
│  Multiple blocks can run on the same SM simultaneously     │
└────────────────────────────────────────────────────────────┘
```

**Key terms:**
- **Grid**: The entire set of work for one kernel launch. Defined by the programmer.
- **Block (Thread Block)**: A group of threads that can cooperate via shared memory. Assigned to one SM.
- **Warp**: 32 threads that execute the **same instruction** at the **same time** (SIMT — Single Instruction, Multiple Threads). This is the fundamental execution unit.
- **Thread**: The smallest unit of execution. Each thread has its own registers and program counter.

#### How a Kernel Launch Maps to Hardware

When you launch a Triton kernel with `kernel[(num_tokens, num_heads)](...)`:

```
1. CPU sends launch command to GPU via PCIe/NVLink
2. GPU's Giga Thread Engine receives the grid specification
3. Grid = num_tokens × num_heads blocks
4. GPU distributes blocks across available SMs:
   - SM 0 gets Block (0,0), Block (0,1), ...
   - SM 1 gets Block (1,0), Block (1,1), ...
   - Each SM can run multiple blocks if it has enough resources
5. Within each SM, blocks are divided into warps of 32 threads
6. Warp schedulers issue instructions to warps every clock cycle
7. When a warp is waiting for memory, the scheduler switches to another warp
   (this is how GPUs hide memory latency — "latency hiding")
```

#### Memory Hierarchy (Speed vs Size)

```
                    Speed           Size        Scope
                    ─────           ────        ─────
Registers:          ~0 cycles       256KB/SM    Per thread
Shared Memory:      ~20 cycles      256KB/SM    Per block (threads cooperate)
L1 Cache:           ~30 cycles      (shared with above)
L2 Cache:           ~200 cycles     50MB        All SMs
Global Memory(HBM): ~400 cycles     80GB        All SMs, all kernels

The key insight: Global memory is ~200× slower than registers.
This is why FUSION matters — keeping data in registers/shared memory
instead of writing to and reading from global memory saves enormous time.
```

#### Tensor Cores: The Matrix Multiply Accelerators

Tensor Cores are specialized hardware units designed for one operation: **matrix multiply-accumulate (MMA)**:

```
D = A × B + C

Where A, B, C, D are small matrix tiles:
- Hopper (H100): 16×16×16 tiles in FP16/BF16, or 16×8×32 in FP8
- One Tensor Core instruction computes this entire tile operation
- This is ~16× faster than doing it with regular CUDA cores
```

Tensor Cores are why modern AI is fast. Every matrix multiplication in a transformer (QKV projections, attention scores, MLP layers) uses Tensor Cores.


Input / Output (Hardware Level):
| | Description |
|---|---|
| **Input** | SASS instructions (GPU binary) + data in HBM (global memory) |
| **Output** | Results written back to HBM, ready for the next kernel or for CPU to read |


### The Kernel Zoo: CUTLASS, cuBLAS, Triton, and More

There are multiple "kernel libraries" and they serve different purposes.

```
┌────────────┬──────────────┬───────────────┬──────────────┬──────────────┐
│            │   cuBLAS     │   CUTLASS     │   Triton     │  Hand-CUDA   │
├────────────┼──────────────┼───────────────┼──────────────┼──────────────┤
│ Written in │ Closed-source│ CUDA C++      │ Python       │ CUDA C++     │
│            │ (NVIDIA)     │ (templates)   │ (@triton.jit)│              │
├────────────┼──────────────┼───────────────┼──────────────┼──────────────┤
│ Best for   │ Dense GEMM   │ Custom GEMM   │ Element-wise │ Anything     │
│            │ (standard    │ (quantized,   │ fusions,     │ (if you have │
│            │ matmul)      │ sparse, fused)│ reductions,  │ the time)    │
│            │              │               │ attention    │              │
├────────────┼──────────────┼───────────────┼──────────────┼──────────────┤
│ Flexibility│ Low (fixed   │ High (C++     │ Medium       │ Maximum      │
│            │ API)         │ templates)    │ (Python DSL) │              │
├────────────┼──────────────┼───────────────┼──────────────┼──────────────┤
│ Performance│ Excellent    │ Excellent     │ Good-Great   │ Depends on   │
│ (matmul)   │ (NVIDIA      │ (can match    │ (90-100% of  │ the author   │
│            │ hand-tuned)  │ cuBLAS)       │ cuBLAS)      │              │
├────────────┼──────────────┼───────────────┼──────────────┼──────────────┤
│ Ease of use│ Easy (just   │ Hard (C++     │ Easy-Medium  │ Very Hard    │
│            │ call API)    │ template      │ (Python-like)│              │
│            │              │ metaprog.)    │              │              │
├────────────┼──────────────┼───────────────┼──────────────┼──────────────┤
│ Compile    │ None (pre-   │ Long (C++     │ Fast (JIT    │ Long (nvcc)  │
│ time       │ compiled)    │ compilation)  │ at runtime)  │              │
└────────────┴──────────────┴───────────────┴──────────────┴──────────────┘
```

#### cuBLAS: NVIDIA's Matrix Multiply Library

**What it is**: A closed-source, pre-compiled library from NVIDIA that implements BLAS (Basic Linear Algebra Subprograms) operations on GPU. The most important operation is GEMM (General Matrix Multiply).

**How it works**:
```python
# From Python/PyTorch:
output = torch.mm(A, B)  # This dispatches to cuBLAS internally

# What happens under the hood:
# 1. PyTorch calls cublasGemmEx() via C++ binding
# 2. cuBLAS selects the best kernel for the given shapes and dtypes
# 3. cuBLAS launches the kernel on the GPU
# 4. The kernel uses Tensor Cores for the actual computation
```

**When to use**: Standard dense matrix multiplications (FP16, BF16, FP32). This is the default for `torch.mm()` and `torch.matmul()`.

**Architecture**:
```
torch.mm(A, B)
    │
    ▼
cuBLAS Runtime (selects best algorithm)
    │
    ├── For small matrices: use CUDA cores
    ├── For large matrices: use Tensor Cores
    │   └── Tiles the matrices into blocks
    │       └── Each SM processes one or more tiles
    │           └── Tensor Cores do 16×16×16 MMA operations
    │               └── Results accumulated in registers
    │                   └── Written back to global memory
    │
    ▼
Output tensor in GPU memory
```

**Input**: Two matrices A (M×K) and B (K×N), data types, algorithm hints
**Output**: Result matrix C (M×N)

#### CUTLASS: NVIDIA's Open-Source GEMM Templates

**What it is**: An open-source C++ template library from NVIDIA for implementing high-performance GEMM and related operations. Unlike cuBLAS (which is a black box), CUTLASS gives you the building blocks to create custom matrix multiply kernels.

**Why it exists**: cuBLAS only supports standard data types and operations. For quantized inference (FP8, INT8, INT4, mixed precision), you need custom kernels. CUTLASS provides the templates to build these.

**How CUTLASS works (conceptual)**:

```
┌─────────────────────────────────────────────────────────────┐
│                    CUTLASS Architecture                     │
│                                                             │
│  Level 1: Device-level                                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Grid of thread blocks, each processing a tile of C     │ │
│  │ C_tile[i,j] = A_tile[i,:] × B_tile[:,j]                │ │
│  └─────────────────────────┬──────────────────────────────┘ │
│                            │                                │
│  Level 2: Thread Block-level (one SM)                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Load A and B tiles from global memory → shared memory  │ │
│  │ Iterate over K dimension in chunks                     │ │
│  │ Each chunk: shared mem → registers → Tensor Core MMA   │ │
│  └─────────────────────────┬──────────────────────────────┘ │
│                            │                                │
│  Level 3: Warp-level                                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Each warp computes a sub-tile of the block's output    │ │
│  │ Uses warp-level matrix multiply (wmma or mma.sync)     │ │
│  └─────────────────────────┬──────────────────────────────┘ │
│                            │                                │
│  Level 4: Tensor Core instruction                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ D = A × B + C  (16×16×16 tile, one clock cycle)        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  Epilogue: Apply post-processing (scaling, bias, activation)│
│  ┌────────────────────────────────────────────────────────┐ │
│  │ output = activation(alpha * D + beta * C + bias)       │ │
│  │ Can also do: quantization, type conversion, etc.       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**How vLLM uses CUTLASS**:

```python
# From vllm/model_executor/kernels/linear/scaled_mm/cutlass.py
class CutlassFP8ScaledMMLinearKernel:
    def apply_scaled_mm(self, *, A, B, As, Bs, out_dtype, bias, ...):
        # Calls into C++ extension that uses CUTLASS templates
        output = ops.cutlass_scaled_mm(
            A, B,                    # FP8 matrices
            out_dtype=out_dtype,     # Output in BF16/FP16
            scale_a=As,              # Per-tensor or per-token scale for A
            scale_b=Bs,              # Per-channel scale for B
            bias=bias,               # Optional bias
        )
        return output
```

The C++ side (`csrc/quantization/cutlass_w8a8/`) instantiates CUTLASS templates for specific configurations:
- FP8 × FP8 → BF16 with per-token/per-channel scaling
- INT8 × INT8 → INT32 with scaling and zero-point adjustment
- FP4 × FP4 with block-wise scaling
- 2:4 structured sparsity

**CUTLASS Kernel Scheduling** (Hopper-specific):
```
TmaWarpSpecialized:            Basic TMA-based data movement
TmaWarpSpecializedPingpong:    Double-buffered for latency hiding
TmaWarpSpecializedCooperative: Multiple thread blocks cooperate on one tile
```

**Input**: Quantized weight matrices, activation tensors, scale factors, bias
**Output**: Dequantized output tensor in the target dtype

#### Triton Kernels (Hand-Written)

Beyond what Inductor generates, vLLM has many **hand-written Triton kernels** for operations that need custom logic:

```python
# Attention kernels (vllm/v1/attention/ops/)
triton_unified_attention.py      # Full attention with paged KV cache
triton_decode_attention.py       # Optimized for decode (single token)
triton_prefill_attention.py      # Optimized for prefill (many tokens)
triton_merge_attn_states.py      # Merge split-KV attention results

# Quantization kernels
# (various in vllm/model_executor/kernels/)

# MoE (Mixture of Experts) kernels
# (vllm/model_executor/layers/fused_moe/)
```

**When to use Triton vs CUTLASS vs cuBLAS**:

| Operation | Best Choice | Why |
|---|---|---|
| Standard FP16/BF16 matmul | cuBLAS | NVIDIA has spent years optimizing this |
| FP8/INT8 quantized matmul | CUTLASS | Custom scaling/dequantization in epilogue |
| FP4/INT4 weight-only matmul | CUTLASS or Marlin | Need custom unpacking logic |
| Attention (paged KV cache) | Triton or FlashAttention | Complex memory access patterns |
| Element-wise fusions (norm+quant) | Triton (via Inductor) | Inductor auto-generates these |
| MoE routing + GEMM | CUTLASS (grouped GEMM) | Need batched GEMM with variable sizes |
| Custom reductions | Triton | Easy to write, good performance |

#### FlashAttention: The Attention Specialist

FlashAttention deserves special mention. It's a hand-optimized attention kernel (originally in CUDA C++, now also in Triton) that:

1. **Tiles** the attention computation to fit in shared memory
2. **Never materializes** the full N×N attention matrix (saves memory)
3. Uses **online softmax** (computes softmax incrementally)
4. Achieves **IO-optimal** memory access patterns

```
Standard Attention:                    FlashAttention:
Q×K^T → S (N×N matrix, huge!)         For each tile of Q:
softmax(S) → P                           For each tile of K,V:
P×V → O                                    Compute partial attention
                                            Update running softmax
Memory: O(N²)                              Accumulate output
                                         Memory: O(N) — no N×N matrix!
```

#### How Kernels Communicate with GPU Computation

The flow from Python to GPU execution:

```
Python Code
    │
    ▼
PyTorch Dispatcher (selects implementation based on dtype, device)
    │
    ├── torch.mm() → cuBLAS (via cublasGemmEx C API)
    ├── ops.cutlass_scaled_mm() → CUTLASS (via C++ extension, torch.ops)
    ├── triton_kernel[grid]() → Triton (JIT compiled, launched via CUDA driver API)
    └── flash_attn() → FlashAttention (via C++ extension)
    │
    ▼
CUDA Driver API
    │
    ├── cuLaunchKernel() — sends kernel + arguments to GPU
    ├── cuMemcpyAsync() — data transfers
    └── cuStreamSynchronize() — wait for completion
    │
    ▼
GPU Hardware
    │
    ├── Giga Thread Engine distributes blocks to SMs
    ├── SMs execute warps using CUDA cores + Tensor Cores
    └── Results written to HBM (global memory)
```

---

### CUDA Graphs: Replaying Entire Workflows

#### What Are CUDA Graphs?

Even after compilation, there's still overhead: for each kernel launch, the CPU must prepare arguments, call the CUDA driver, and wait for the GPU to pick up the work. For a transformer with 32 layers, each with ~10 kernel launches, that's 320 CPU→GPU round trips per forward pass.

**CUDA Graphs** solve this by recording an entire sequence of kernel launches once, then replaying the whole sequence with a single CPU command.

```
WITHOUT CUDA Graphs:                    WITH CUDA Graphs:
─────────────────────                   ─────────────────
CPU: launch kernel 1                    CPU: replay graph (1 command)
CPU: launch kernel 2                    GPU: kernel 1 → kernel 2 → ... → kernel N
CPU: launch kernel 3                    
...                                     Total CPU overhead: ~1 launch
CPU: launch kernel N                    
                                        
Total CPU overhead: N launches          
(each ~5-10 μs)                         
```

#### How CUDA Graphs Work

```
Phase 1: CAPTURE (done once at startup)
┌─────────────────────────────────────────────────────┐
│  cudaStreamBeginCapture(stream)                     │
│                                                     │
│  // Run the model forward pass normally             │
│  kernel_1<<<grid, block, 0, stream>>>(args...)      │
│  kernel_2<<<grid, block, 0, stream>>>(args...)      │
│  ...                                                │
│  kernel_N<<<grid, block, 0, stream>>>(args...)      │
│                                                     │
│  cudaStreamEndCapture(stream, &graph)               │
│  cudaGraphInstantiate(&graphExec, graph)            │
│                                                     │
│  // Nothing actually ran on GPU — just recorded!    │
└─────────────────────────────────────────────────────┘

Phase 2: REPLAY (done for every inference request)
┌─────────────────────────────────────────────────────┐
│  // Copy new input data into the pre-allocated      │
│  // input buffers (same memory addresses as capture)│
│  cudaMemcpy(input_buffer, new_data, ...)            │
│                                                     │
│  // Replay the entire graph with ONE command        │
│  cudaGraphLaunch(graphExec, stream)                 │
│                                                     │
│  // Read output from pre-allocated output buffer    │
└─────────────────────────────────────────────────────┘
```

#### vLLM's CUDA Graph Modes

vLLM supports multiple CUDA graph strategies:

| Mode | What It Does | Best For |
|---|---|---|
| `NONE` | No CUDA graphs (pure eager) | Debugging |
| `PIECEWISE` | Captures graphs between attention ops, attention runs eager | Most flexible, works with all attention backends |
| `FULL` | Captures the entire forward pass including attention | Maximum performance, requires compatible attention backend |
| `FULL_DECODE_ONLY` | Full graphs for decode, no graphs for prefill | Prefill/decode disaggregation |
| `FULL_AND_PIECEWISE` | Full for decode, piecewise for prefill | Best overall performance (default) |

The piecewise approach is clever:

```
┌──────────────┐  ┌───────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────────┐
│ CUDA Graph 0 │  │ Attention │  │ CUDA Graph 1 │  │ Attention │  │ CUDA Graph 2 │
│ (norm+proj)  │→ │ (eager)   │→ │ (proj+norm+  │→ │ (eager)   │→ │ (proj+norm)  │
│              │  │           │  │  MLP)        │  │           │  │              │
└──────────────┘  └───────────┘  └──────────────┘  └───────────┘  └──────────────┘
     Replayed        Launched         Replayed         Launched        Replayed
     (fast)          (flexible)       (fast)           (flexible)      (fast)
```

### How vLLM Puts It All Together

####  The Complete Compilation Pipeline in vLLM

```
┌─────────────────────────────────────────────────────────────────────┐
│                    vLLM Server Startup                              │
│                                                                     │
│  Step 1: Load Model                                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Load model weights from disk/HuggingFace                       │ │
│  │ Initialize model as standard PyTorch nn.Module                 │ │
│  │ CustomOp dispatch: choose CUDA/Triton/cuBLAS per operation     │ │
│  └──────────────────────────────┬─────────────────────────────────┘ │
│                                 │                                   │
│  Step 2: torch.compile (Dynamo + Inductor)                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ a) Dynamo traces model.forward() → FX Graph                    │ │
│  │ b) Wrap attention as opaque custom op (no graph breaks)        │ │
│  │ c) Split FX graph at attention boundaries                      │ │
│  │ d) For each piece:                                             │ │
│  │    - Apply vLLM custom passes (fusion, SP, async TP)           │ │
│  │    - Inductor generates Triton kernels                         │ │
│  │    - Cache compiled artifacts to disk                          │ │
│  │ e) Attention pieces use pre-existing optimized kernels         │ │
│  │    (FlashAttention, FlashInfer, Triton attention, etc.)        │ │
│  └──────────────────────────────┬─────────────────────────────────┘ │
│                                 │                                   │
│  Step 3: CUDA Graph Capture                                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ For each batch size in cudagraph_capture_sizes:                │ │
│  │   a) Warm up with dummy data (eager mode)                      │ │
│  │   b) Capture CUDA graph (piecewise or full)                    │ │
│  │   c) Store graph keyed by BatchDescriptor                      │ │
│  └──────────────────────────────┬─────────────────────────────────┘ │
│                                 │                                   │
│  Step 4: Ready to Serve                                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ All compilation is done BEFORE any request is served           │ │
│  │ No request will ever trigger recompilation                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

#### Runtime Inference Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Request Arrives                                  │
│                                                                     │
│  1. Scheduler batches requests, determines batch_size               │
│                                                                     │
│  2. CudagraphDispatcher.dispatch(batch_descriptor)                  │
│     ├── Decode batch? → Try FULL mode first                         │
│     ├── Prefill/mixed? → Try PIECEWISE mode                         │
│     └── No matching graph? → Fall back to NONE (eager)              │
│                                                                     │
│  3. Execute model forward:                                          │
│     ┌─────────────────────────────────────────────────────────────┐ │
│     │ IF CUDA graph exists for this batch:                        │ │
│     │   Copy inputs to static buffers → Replay graph → Read output│ │
│     │ ELSE:                                                       │ │
│     │   Run compiled kernels directly (still fast, just no graph) │ │
│     └─────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  4. Inside the model forward (per layer):                           │
│     ┌─────────────────────────────────────────────────────────────┐ │
│     │ Piece 0 (Inductor-compiled Triton kernel):                  │ │
│     │   RMSNorm → QKV projection (cuBLAS or CUTLASS)              │ │
│     │                                                             │ │
│     │ Attention (custom op, not compiled by Inductor):            │ │
│     │   FlashAttention / FlashInfer / Triton attention kernel     │ │
│     │   Interacts with paged KV cache                             │ │
│     │                                                             │ │
│     │ Piece 1 (Inductor-compiled Triton kernel):                  │ │
│     │   Output projection → Residual → RMSNorm → MLP              │ │
│     │   (SiLU+Mul may be fused with FP8 quantization)             │ │
│     │                                                             │ │
│     │ Attention (custom op):                                      │ │
│     │   ... same as above ...                                     │ │
│     │                                                             │ │
│     │ Piece 2 (Inductor-compiled Triton kernel):                  │ │
│     │   Down projection → Residual                                │ │
│     └─────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  5. LM Head + Sampling (outside compiled graph)                     │
│     Token probabilities → Sample next token → Return to user        │
└─────────────────────────────────────────────────────────────────────┘
```

#### vLLM's Compilation Modes

```python
class CompilationMode(enum.IntEnum):
    NONE = 0              # Pure eager PyTorch — no compilation at all
    STOCK_TORCH_COMPILE = 1  # Standard torch.compile (Dynamo + Inductor)
    DYNAMO_TRACE_ONCE = 2    # Trace once, avoid recompilation
    VLLM_COMPILE = 3         # Full vLLM pipeline: custom passes + piecewise
                             # compilation + CUDA graphs (DEFAULT in V1)
```

#### vLLM's Kernel Selection Strategy

vLLM dynamically selects the best kernel implementation based on hardware and quantization:

```python
# From vllm/model_executor/kernels/linear/__init__.py
# Priority order for FP8 kernels on CUDA:
_POSSIBLE_FP8_KERNELS = {
    PlatformEnum.CUDA: [
        FlashInferFP8ScaledMMLinearKernel,   # 1st choice: FlashInfer
        CutlassFP8ScaledMMLinearKernel,      # 2nd choice: CUTLASS
        PerTensorTorchFP8ScaledMMLinearKernel, # 3rd: PyTorch native (cuBLAS)
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
}

# For INT8:
_POSSIBLE_INT8_KERNELS = {
    PlatformEnum.CUDA: [
        CutlassInt8ScaledMMLinearKernel,     # 1st choice: CUTLASS
        TritonInt8ScaledMMLinearKernel,       # 2nd choice: Triton
    ],
}

# For mixed-precision (W4A16, W8A16):
_POSSIBLE_KERNELS = {
    PlatformEnum.CUDA: [
        CutlassW4A8LinearKernel,   # CUTLASS for W4A8
        MacheteLinearKernel,       # Machete (CUTLASS-based, for Hopper)
        MarlinLinearKernel,        # Marlin (hand-optimized CUDA)
        ExllamaLinearKernel,       # Exllama (GPTQ)
    ],
}
```

#### The CustomOp System

vLLM's `CustomOp` base class provides platform-aware dispatch:

```python
class SiluAndMul(CustomOp):
    """When compiled with Inductor: Inductor generates a fused Triton kernel.
       When running with custom ops enabled: dispatches to hand-written CUDA kernel."""
    
    def forward_native(self, x):
        """PyTorch-native implementation (used by Inductor for fusion)"""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]
    
    def forward_cuda(self, x):
        """Hand-written CUDA kernel (faster, but opaque to compiler)"""
        d = x.shape[-1] // 2
        output = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
        ops.silu_and_mul(output, x)  # C++ extension
        return output
```

The key insight: when `torch.compile` is active with Inductor backend, custom ops are **disabled by default**. This lets Inductor see the `forward_native` implementation and fuse it with surrounding operations (e.g., fusing SiLU+Mul with FP8 quantization). When running without compilation, the hand-written CUDA kernels are used instead.

## Glossary

| Term | Definition |
|---|---|
| **BLAS** | Basic Linear Algebra Subprograms — standard API for matrix operations |
| **cuBLAS** | NVIDIA's GPU implementation of BLAS (closed-source, highly optimized) |
| **CUTLASS** | CUDA Templates for Linear Algebra Subroutines — NVIDIA's open-source C++ template library for custom GEMM kernels |
| **CUDA Core** | General-purpose floating-point/integer compute unit on an SM |
| **CUDA Graph** | A recorded sequence of GPU operations that can be replayed with minimal CPU overhead |
| **Dynamo (TorchDynamo)** | Python bytecode interceptor that captures computation graphs from PyTorch code |
| **Eager Mode** | Default PyTorch execution: operations run one at a time as Python encounters them |
| **Epilogue** | Post-processing step in a GEMM kernel (scaling, bias, activation, quantization) |
| **FlashAttention** | IO-aware attention algorithm that tiles computation to avoid materializing the N×N attention matrix |
| **FX Graph** | PyTorch's intermediate representation for computation graphs |
| **Fusion** | Combining multiple operations into a single kernel to reduce memory traffic |
| **GEMM** | General Matrix Multiply: C = α·A·B + β·C |
| **Grid** | The set of all thread blocks launched for a kernel |
| **Guard** | A condition that must be true for a compiled graph to be valid (e.g., input shape) |
| **HBM** | High Bandwidth Memory — the main GPU memory (large but relatively slow) |
| **Inductor** | PyTorch's default compiler backend that generates Triton/C++ code from FX graphs |
| **Kernel** | A function that runs on the GPU, executed by many threads in parallel |
| **MMA** | Matrix Multiply-Accumulate — the operation performed by Tensor Cores |
| **Piecewise Compilation** | vLLM's strategy of splitting the graph at attention boundaries and compiling each piece independently |
| **PTX** | Parallel Thread Execution — NVIDIA's virtual GPU assembly language |
| **SASS** | Shader Assembly — the actual machine code for a specific GPU architecture |
| **SM (Streaming Multiprocessor)** | The fundamental compute unit of an NVIDIA GPU, containing CUDA cores, Tensor Cores, shared memory, and schedulers |
| **Shared Memory** | Fast on-chip memory (SRAM) shared by all threads in a block (~20 cycle latency) |
| **Tensor Core** | Specialized hardware unit for matrix multiply-accumulate operations (16×16×16 tiles) |
| **TMA** | Tensor Memory Accelerator — Hopper hardware unit for efficient bulk data movement |
| **Triton** | Python-based GPU kernel language that compiles to PTX/SASS via MLIR and LLVM |
| **Warp** | A group of 32 threads that execute the same instruction simultaneously (SIMT) |

---

## Appendix A: Additional Topics Worth Understanding

### A.1 Memory Bandwidth vs Compute Bound

Most LLM inference operations are **memory-bandwidth bound**, not compute bound:

```
Operation          Arithmetic Intensity    Bottleneck
─────────          ────────────────────    ──────────
Element-wise       Very low (1 FLOP/byte)  Memory bandwidth
(ReLU, add, norm)

Attention          Medium                  Memory (decode) / Compute (prefill)

Matrix Multiply    High (O(N) FLOPs/byte)  Compute (large batch)
                                           Memory (small batch / decode)
```

This is why fusion matters so much for element-wise operations — you're saving memory reads/writes, which is the actual bottleneck.

### A.2 Quantization and Kernel Selection

Modern LLM serving uses quantized weights to reduce memory usage and increase throughput:

```
Precision    Bits/Weight    Memory Savings    Kernel Used
─────────    ───────────    ──────────────    ───────────
FP32         32 bits        1× (baseline)     cuBLAS
FP16/BF16    16 bits        2×                cuBLAS
FP8          8 bits         4×                CUTLASS / FlashInfer
INT8         8 bits         4×                CUTLASS / Triton
INT4/FP4     4 bits         8×                CUTLASS / Marlin / Machete
```

Each quantization format needs specialized kernels because:
1. The data must be **dequantized** (converted back to higher precision) during computation
2. **Scale factors** must be applied (per-tensor, per-channel, per-token, or per-block)
3. The dequantization should happen **inside the kernel** (fused with the GEMM) to avoid extra memory traffic

### A.3 The Roofline Model

The roofline model helps you understand whether a kernel is limited by compute or memory:

```
Performance │                    ╱ Compute Ceiling
(FLOPS)     │                  ╱─────────────────
            │                ╱
            │              ╱
            │            ╱
            │          ╱
            │        ╱
            │      ╱  ← Memory Bandwidth Ceiling
            │    ╱
            │  ╱
            │╱
            └──────────────────────────────────
              Arithmetic Intensity (FLOPS/byte)
              
Low intensity (left): Memory bound → fusion helps
High intensity (right): Compute bound → Tensor Cores help
```

### A.4 How to Profile and Debug

Tools for understanding what's happening on the GPU:

| Tool | What It Shows |
|---|---|
| `torch.profiler` | Python-level profiling with GPU kernel timing |
| `nsight systems` | Timeline view of CPU/GPU activity, kernel launches, memory transfers |
| `nsight compute` | Deep dive into a single kernel: SM utilization, memory throughput, warp stalls |
| `TORCH_COMPILE_DEBUG=1` | Dumps Inductor's generated code and optimization decisions |
| `VLLM_LOGGING_LEVEL=DEBUG` | Shows vLLM's compilation cache, graph splitting, kernel selection |

---

*This document covers the full stack from Python to GPU silicon. For hands-on implementation, start by reading the generated Inductor code in `~/.cache/vllm/torch_compile_cache/` and the Triton kernels in `vllm/v1/attention/ops/`. For CUTLASS, start with the C++ extensions in `csrc/quantization/`.*