import importlib
import torch

_triton_mod = importlib.import_module("llm.compilation.04_triton_kernels")
vector_add_kernel = _triton_mod.vector_add_kernel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("\n==== PTX / SASS — Inspecting GPU Assembly ====\n")

print("""
Every Triton kernel compiles down to PTX (virtual assembly) and then SASS (real machine code). Let's inspect the PTX for vector_add kernel.

PTX — Parallel Thread eXecution. It's NVIDIA's virtual ISA (instruction set architecture), a stable intermediate 
representation that sits between Triton/CUDA C and actual GPU hardware. Think of it like bytecode — it's not tied 
to a specific GPU generation.

SASS — Shader ASSembly. This is the actual machine code that runs on the GPU hardware. The NVIDIA driver compiles 
PTX → SASS at runtime for the specific GPU you're running on (e.g. sm_80 for A100, sm_89 for RTX 4090).

The compilation chain is: Python/Triton → PTX → SASS → GPU execution
PTX is portable across GPU generations; SASS is not. 
That's why NVIDIA ships PTX in CUDA binaries — if you run old code on a new GPU, the driver JIT-compiles the PTX to the new GPU's SASS.
""")

print("\n==== Extracting PTX from a Triton Kernel ====\n") 

# Compile the kernel to get the PTX
# We need to call the kernel once to trigger compilation
a_small = torch.randn(1024, device=DEVICE, dtype=torch.float32)
b_small = torch.randn(1024, device=DEVICE, dtype=torch.float32)
out_small = torch.empty(1024, device=DEVICE, dtype=torch.float32)

# Launch to trigger compilation
vector_add_kernel[(1,)](a_small, b_small, out_small, 1024, BLOCK_SIZE=1024)

# Try to extract the compiled kernel info
try:
    import triton, glob, os
    print(f"Triton version: {triton.__version__}")

    # Triton 3.x stores kernel artifacts (ptx, ttir, llir) in per-kernel cache dirs, not the launcher dir. 
    ptx_files = glob.glob(os.path.expanduser("~/.triton/cache/**/vector_add_kernel.ptx"), recursive=True)

    if ptx_files:
        ptx_file = ptx_files[0]
        print(f"Found PTX: {ptx_file}\n")
        with open(ptx_file) as f:
            lines = f.readlines()
        print(f"PTX code ({len(lines)} lines total), showing first 40:\n")
        for line in lines[:40]:
            print(f"  {line}", end="")
        print(f"\n  ... ({len(lines) - 40} more lines)")

        # Also show the Triton IR for comparison
        ttir_file = ptx_file.replace(".ptx", ".ttir")
        if os.path.exists(ttir_file):
            with open(ttir_file) as f:
                ttir_lines = f.readlines()
            print(f"\nTriton IR (.ttir) — {len(ttir_lines)} lines, showing first 20:\n")
            for line in ttir_lines[:20]:
                print(f"  {line}", end="")
    else:
        print("  No vector_add_kernel.ptx found — run the kernel first")
except Exception as e:
    print(f"  (Could not extract PTX: {e})")


print("\n==== Understanding PTX Instructions ====\n") 

print("""
Key PTX instructions you'll see in GPU kernels:

  MEMORY:
    ld.global.v4.f32  %f1, [%rd1]     Load 4 floats from global memory
    st.global.f32     [%rd2], %f5      Store 1 float to global memory
    ld.shared.f32     %f3, [%r1]       Load from shared memory (fast!)

  ARITHMETIC:
    add.f32           %f4, %f1, %f2    Float addition
    mul.f32           %f5, %f3, %f4    Float multiplication
    fma.rn.f32        %f6, %f1, %f2, %f3   Fused multiply-add: f1*f2+f3

  CONTROL:
    @%p1 bra          LABEL            Conditional branch (predicated)
    bar.sync          0                Synchronize threads in a block

  SPECIAL:
    ex2.approx.f32    %f7, %f6         Fast 2^x (used for exp, sigmoid)
    rsqrt.approx.f32  %f8, %f7         Fast 1/sqrt(x) (used in LayerNorm)

  TENSOR CORE:
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {%f0,%f1,%f2,%f3}, {%f4,%f5,...}, {%f8,%f9,...}, {%f12,...}
        → Matrix multiply-accumulate: D = A × B + C
        → Processes a 16×8×16 tile in ONE instruction
""")


"""
python -m llm.compilation.05_ptx_inspection

==== PTX / SASS — Inspecting GPU Assembly ====

Every Triton kernel compiles down to PTX (virtual assembly) and then SASS (real machine code). Let's inspect the PTX for vector_add kernel.

PTX — Parallel Thread eXecution. It's NVIDIA's virtual ISA (instruction set architecture), a stable intermediate 
representation that sits between Triton/CUDA C and actual GPU hardware. Think of it like bytecode — it's not tied 
to a specific GPU generation.

SASS — Shader ASSembly. This is the actual machine code that runs on the GPU hardware. The NVIDIA driver compiles 
PTX → SASS at runtime for the specific GPU you're running on (e.g. sm_80 for A100, sm_89 for RTX 4090).

The compilation chain is: Python/Triton → PTX → SASS → GPU execution
PTX is portable across GPU generations; SASS is not. 
That's why NVIDIA ships PTX in CUDA binaries — if you run old code on a new GPU, the driver JIT-compiles the PTX to the new GPU's SASS.

==== Extracting PTX from a Triton Kernel ====

Triton version: 3.5.1
Found PTX: /home/ubuntu/.triton/cache/ERVNVOGAO4B42XFIIDQDWWIJFN7IICDTARYXITUMHA7EPW5YIISA/vector_add_kernel.ptx

PTX code (210 lines total), showing first 40:

  //
  // Generated by LLVM NVPTX Back-End
  //
  
  .version 8.7
  .target sm_90a
  .address_size 64
  
        // .globl       vector_add_kernel       // -- Begin function vector_add_kernel
                                          // @vector_add_kernel
  .visible .entry vector_add_kernel(
        .param .u64 .ptr .global .align 1 vector_add_kernel_param_0,
        .param .u64 .ptr .global .align 1 vector_add_kernel_param_1,
        .param .u64 .ptr .global .align 1 vector_add_kernel_param_2,
        .param .u32 vector_add_kernel_param_3,
        .param .u64 .ptr .global .align 1 vector_add_kernel_param_4,
        .param .u64 .ptr .global .align 1 vector_add_kernel_param_5
  )
  .reqntid 128
  {
        .reg .pred      %p<7>;
        .reg .b32       %r<33>;
        .reg .b64       %rd<11>;
        .loc    1 24 0                          // 04_triton_kernels.py:24:0
  $L__func_begin0:
        .loc    1 24 0                          // 04_triton_kernels.py:24:0
  
  // %bb.0:
        ld.param.b64    %rd7, [vector_add_kernel_param_0];
        ld.param.b64    %rd8, [vector_add_kernel_param_1];
  $L__tmp0:
        .loc    1 38 24                         // 04_triton_kernels.py:38:24
        mov.u32         %r25, %ctaid.x;
        .loc    1 41 24                         // 04_triton_kernels.py:41:24
        shl.b32         %r26, %r25, 10;
        ld.param.b64    %rd9, [vector_add_kernel_param_2];
        ld.param.b32    %r27, [vector_add_kernel_param_3];
        .loc    1 42 41                         // 04_triton_kernels.py:42:41
        mov.u32         %r28, %tid.x;
        shl.b32         %r29, %r28, 2;

  ... (170 more lines)

Triton IR (.ttir) — 52 lines, showing first 20:

  #loc = loc("/home/ubuntu/github/til/llm/compilation/04_triton_kernels.py":24:0)
  #loc15 = loc("a_ptr"(#loc))
  #loc16 = loc("b_ptr"(#loc))
  #loc17 = loc("output_ptr"(#loc))
  #loc18 = loc("n_elements"(#loc))
  module {
    tt.func public @vector_add_kernel(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("a_ptr"(#loc)), %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("b_ptr"(#loc)), %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("output_ptr"(#loc)), %n_elements: i32 {tt.divisibility = 16 : i32} loc("n_elements"(#loc))) attributes {noinline = false} {
      %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
      %pid = tt.get_program_id x : i32 loc(#loc19)
      %block_start = arith.muli %pid, %c1024_i32 : i32 loc(#loc20)
      %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc21)
      %offsets_0 = tt.splat %block_start : i32 -> tensor<1024xi32> loc(#loc22)
      %offsets_1 = arith.addi %offsets_0, %offsets : tensor<1024xi32> loc(#loc22)
      %mask = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc23)
      %mask_2 = arith.cmpi slt, %offsets_1, %mask : tensor<1024xi32> loc(#loc23)
      %a = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc24)
      %a_3 = tt.addptr %a, %offsets_1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc24)
      %a_4 = tt.load %a_3, %mask_2 : tensor<1024x!tt.ptr<f32>> loc(#loc25)
      %b = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc26)
      %b_5 = tt.addptr %b, %offsets_1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc26)

==== Understanding PTX Instructions ====


Key PTX instructions you'll see in GPU kernels:

  MEMORY:
    ld.global.v4.f32  %f1, [%rd1]     Load 4 floats from global memory
    st.global.f32     [%rd2], %f5      Store 1 float to global memory
    ld.shared.f32     %f3, [%r1]       Load from shared memory (fast!)

  ARITHMETIC:
    add.f32           %f4, %f1, %f2    Float addition
    mul.f32           %f5, %f3, %f4    Float multiplication
    fma.rn.f32        %f6, %f1, %f2, %f3   Fused multiply-add: f1*f2+f3

  CONTROL:
    @%p1 bra          LABEL            Conditional branch (predicated)
    bar.sync          0                Synchronize threads in a block

  SPECIAL:
    ex2.approx.f32    %f7, %f6         Fast 2^x (used for exp, sigmoid)
    rsqrt.approx.f32  %f8, %f7         Fast 1/sqrt(x) (used in LayerNorm)

  TENSOR CORE:
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {%f0,%f1,%f2,%f3}, {%f4,%f5,...}, {%f8,%f9,...}, {%f12,...}
        → Matrix multiply-accumulate: D = A × B + C
        → Processes a 16×8×16 tile in ONE instruction
"""