import os
import time
import torch

from llm.mini_transformer import MiniFfnBlock, MiniTransformer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("""
INDUCTOR — Graph to Kernel Code Generation

Inductor is the default torch.compile backend. It takes the FX graph and generates optimized Triton kernels (or C++ code).

Let's compile our model and inspect what Inductor produces.
""")

print("\n==== Compiling with Inductor and Inspecting Output ====\n")

# Enable debug output to see generated code
os.environ["TORCH_COMPILE_DEBUG"] = "0"  # Set to "1" for VERY verbose output

model_for_compile = MiniFfnBlock(1024).to(DEVICE).half()

# Compile with inductor backend
compiled_model = torch.compile(
    model_for_compile,
    backend="inductor",
    mode="reduce-overhead",  # Enables CUDA graphs too
)

# First call triggers compilation
print("Triggering compilation (first call)...")
compile_input = torch.randn(2, 64, 1024, device=DEVICE, dtype=torch.float16)
start = time.perf_counter()
_ = compiled_model(compile_input)
torch.cuda.synchronize()
compile_time = ( time.perf_counter() - start ) * 1000
print(f"First call (includes compilation): {compile_time:.2f} ms")

# Second call uses cached compiled code
start = time.perf_counter()
for _ in range(10):
    _ = compiled_model(compile_input)
torch.cuda.synchronize()
cached_time = (time.perf_counter() - start) * 1000 / 10
print(f"Subsequent calls: {cached_time:.3f} ms")

print("\n==== What Inductor Actually Generates====\n")

print("""
Inductor generates Triton kernel code. Here's what a fused kernel
looks like conceptually (simplified from actual Inductor output):

    @triton.jit
    def fused_layer_norm_silu_kernel(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, eps, BLOCK_SIZE: tl.constexpr
    ):
        # 1. Load a block of input data
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)

        # 2. Compute layer norm (mean, variance, normalize)
        x = tl.load(input_ptr + row_idx * N + col_offsets)
        mean = tl.sum(x) / N
        var = tl.sum((x - mean) ** 2) / N
        x_norm = (x - mean) / tl.sqrt(var + eps)

        # 3. Apply SiLU activation (FUSED — no memory round-trip!)
        x_silu = x_norm * tl.sigmoid(x_norm)

        # 4. Store result
        tl.store(output_ptr + row_idx * N + col_offsets, x_silu)

Key insight: LayerNorm + SiLU that would be 2 separate kernels in eager
mode become 1 kernel. Data stays in registers, never written to HBM
between operations.
""")

print("\n==== Seeing the Fusion in Action ====\n")

print("Let's use a custom backend to see exactly what Inductor receives:\n")

fusion_graphs = []

def inspection_backend(gm, example_inputs):
    """Custom backend that prints the graph then delegates to Inductor."""
    fusion_graphs.append(gm)
    node_count = len([n for n in gm.graph.nodes
                      if n.op == "call_function"])
    print(f"  Graph received by backend: {node_count} function call nodes")

    # Show the operations
    ops_seen = []
    for node in gm.graph.nodes:
        if node.op == "call_function":
            name = str(node.target).split(".")[-1]
            if name not in ops_seen:
                ops_seen.append(name)
    print(f"  Unique operations: {', '.join(ops_seen[:15])}")
    if len(ops_seen) > 15:
        print(f"    ... and {len(ops_seen) - 15} more")

    # Delegate to inductor for actual compilation
    from torch._inductor.compile_fx import compile_fx
    return compile_fx(gm, example_inputs)

model_inspect = MiniFfnBlock(256).to(DEVICE).half()
compiled_inspect = torch.compile(model_inspect, backend=inspection_backend)

inspect_input = torch.randn(2, 8, 256, device=DEVICE, dtype=torch.float16)
print("Compiling model (inspection backend):")
_ = compiled_inspect(inspect_input)
torch.cuda.synchronize()

print("\n==== Compilation Speedup Measurement ====\n")

print("""
Inductor introduces more overhead for MiniFfnBlock as The GPU finishes the eager kernels so fast that 
the overhead of CUDA graph management and Triton kernel dispatch in compiled mode actually dominates.

Inductor's benefits (kernel fusion, reduced memory bandwidth) only pay off when:
- The model is compute-bound (not bandwidth-bound)
- There are many ops to fuse (Mini FFN only has ~6 ops)
- The batch/sequence dimensions are large enough to saturate the GPU
""")

def benchmark(hidden_size, compile_mode):
    print(f"--- hidden_size={hidden_size}, compile_mode={compile_mode!r} ---")
    model_eager = MiniFfnBlock(hidden_size).to(DEVICE).half()
    model_bench = MiniFfnBlock(hidden_size).to(DEVICE).half()
    
    DEVICE_1 = "cuda:1" if torch.cuda.is_available() else "cpu"
    transformer_eager = MiniTransformer(vocab_size=32000, hidden_size=hidden_size, num_layers=12).to(DEVICE_1).half()
    transformer_bench = MiniTransformer(vocab_size=32000, hidden_size=hidden_size, num_layers=12).to(DEVICE_1).half()

    compiled_bench = torch.compile(model_bench, backend="inductor", mode=compile_mode)
    compiled_transformer_bench = torch.compile(transformer_bench, backend="inductor", mode=compile_mode)

    bench_input = torch.randn(8, 128, hidden_size, device=DEVICE, dtype=torch.float16)
    transformer_bench_input = torch.randint(0, 32000, (4, 512), device=DEVICE_1)

    # Warm up both
    for _ in range(20):
        _ = model_eager(bench_input)
        _ = compiled_bench(bench_input)
        _ = transformer_eager(transformer_bench_input)
        _ = compiled_transformer_bench(transformer_bench_input)
    torch.cuda.synchronize()

    # Benchmark eager
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = model_eager(bench_input)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - start) / 50 * 1000

    # Benchmark compiled
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = compiled_bench(bench_input)
    torch.cuda.synchronize()
    compiled_ms = (time.perf_counter() - start) / 50 * 1000

    speedup = eager_ms / compiled_ms
    print(f"Eager mode:    {eager_ms:.3f} ms")
    print(f"Compiled mode: {compiled_ms:.3f} ms")
    print(f"FFN Speedup:       {speedup:.2f}×\n")

    # Benchmark transformer eager
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = transformer_eager(transformer_bench_input)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - start) / 50 * 1000

    # Benchmark transformer compiled
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = compiled_transformer_bench(transformer_bench_input)
    torch.cuda.synchronize()
    compiled_ms = (time.perf_counter() - start) / 50 * 1000

    speedup = eager_ms / compiled_ms
    print(f"Eager mode:    {eager_ms:.3f} ms")
    print(f"Compiled mode: {compiled_ms:.3f} ms")
    print(f"Transformer Speedup:       {speedup:.2f}×\n")

# (1) baseline
benchmark(hidden_size=1024, compile_mode=None)
# (2) reduce-overhead mode
benchmark(hidden_size=1024, compile_mode="reduce-overhead")
# (3) larger hidden size
benchmark(hidden_size=4096, compile_mode=None)
# (4) larger hidden size + reduce-overhead mode
benchmark(hidden_size=4096, compile_mode="reduce-overhead")

print("The speedup comes from kernel fusion and reduced launch overhead.")

"""
python -m llm.compilation.03_inductor

INDUCTOR — Graph to Kernel Code Generation

Inductor is the default torch.compile backend. It takes the FX graph and generates optimized Triton kernels (or C++ code).

Let's compile our model and inspect what Inductor produces.

==== Compiling with Inductor and Inspecting Output ====

Triggering compilation (first call)...
First call (includes compilation): 1300.51 ms
Subsequent calls: 2.975 ms

==== What Inductor Actually Generates====

Inductor generates Triton kernel code. Here's what a fused kernel
looks like conceptually (simplified from actual Inductor output):

    @triton.jit
    def fused_layer_norm_silu_kernel(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, eps, BLOCK_SIZE: tl.constexpr
    ):
        # 1. Load a block of input data
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)

        # 2. Compute layer norm (mean, variance, normalize)
        x = tl.load(input_ptr + row_idx * N + col_offsets)
        mean = tl.sum(x) / N
        var = tl.sum((x - mean) ** 2) / N
        x_norm = (x - mean) / tl.sqrt(var + eps)

        # 3. Apply SiLU activation (FUSED — no memory round-trip!)
        x_silu = x_norm * tl.sigmoid(x_norm)

        # 4. Store result
        tl.store(output_ptr + row_idx * N + col_offsets, x_silu)

Key insight: LayerNorm + SiLU that would be 2 separate kernels in eager
mode become 1 kernel. Data stays in registers, never written to HBM
between operations.

==== Seeing the Fusion in Action ====

Let's use a custom backend to see exactly what Inductor receives:

Compiling model (inspection backend):
  Graph received by backend: 5 function call nodes
  Unique operations: <function layer_norm at 0x75fd9333ee80>, <built-in function linear>, <function silu at 0x75fd9333e840>, <built-in function add>

==== Compilation Speedup Measurement ====

Inductor introduces more overhead for MiniFfnBlock as The GPU finishes the eager kernels so fast that 
the overhead of CUDA graph management and Triton kernel dispatch in compiled mode actually dominates.

Inductor's benefits (kernel fusion, reduced memory bandwidth) only pay off when:
- The model is compute-bound (not bandwidth-bound)
- There are many ops to fuse (Mini FFN only has ~6 ops)
- The batch/sequence dimensions are large enough to saturate the GPU

--- hidden_size=1024, compile_mode=None ---
Eager mode:    0.087 ms
Compiled mode: 0.194 ms
FFN Speedup:               0.45×

Eager mode:    5.254 ms
Compiled mode: 3.140 ms
Transformer Speedup:       1.67×

--- hidden_size=1024, compile_mode='reduce-overhead' ---
Eager mode:    0.092 ms
Compiled mode: 1.744 ms
FFN Speedup:               0.05×

Eager mode:    5.586 ms
Compiled mode: 0.653 ms
Transformer Speedup:       8.55×

--- hidden_size=4096, compile_mode=None ---
Eager mode:    0.398 ms
Compiled mode: 0.405 ms
FFN Speedup:               0.98×

Eager mode:    19.331 ms
Compiled mode: 18.029 ms
Transformer Speedup:       1.07×

--- hidden_size=4096, compile_mode='reduce-overhead' ---
Eager mode:    0.396 ms
Compiled mode: 2.054 ms
FFN Speedup:               0.19×

Eager mode:    18.325 ms
Compiled mode: 6.423 ms
Transformer Speedup:       2.85×

The speedup comes from kernel fusion and reduced launch overhead.
"""