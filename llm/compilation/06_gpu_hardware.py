import time
import torch 

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("\n==== GPU HARDWARE — Measuring What the Silicon Does ====\n")

print("""
Let's measure actual GPU hardware characteristics to understand the performance constraints kernels operate under.
""")

print("\n==== Memory Bandwidth Measurement ====\n")

print("""
GPU memory:
Registers       ~256KB per SM    Fastest, per-thread
L1 / Shared     ~128-256KB per SM  Programmer-controlled (shared mem) or automatic (L1)
L2              ~40-80MB total   Shared across all SMs
HBM             ~40-80GB         Main GPU memory, slowest

L1 is per-SM (Streaming Multiprocessor) and is actually split into two parts on NVIDIA GPUs:
- - Automatic L1 cache — caches global memory reads transparently
Shared memory — explicitly managed by the programmer in CUDA/Triton (tl.load into shared). This is the fast scratchpad you control directly.
L2 is shared across all SMs on the chip. Much larger than L1 but slower.
""")

print("Measuring GPU memory bandwidth (the #1 bottleneck for LLM inference):\n")

# Measure memory bandwidth with a simple copy
sizes_mb = [1, 10, 100, 500]
for size_mb in sizes_mb:
    n_elements = size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
    src = torch.randn(n_elements, device=DEVICE, dtype=torch.float32)
    dst = torch.empty_like(src)

    # Warm up
    dst.copy_(src)
    torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    n_iters = max(10, 1000 // size_mb)
    for _ in range(n_iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters

    bytes_moved = n_elements * 4 * 2  # read + write
    bandwidth_gb_s = bytes_moved / elapsed / 1e9
    print(f"  {size_mb:>4d} MB copy: {elapsed*1000:.3f} ms, "
          f"bandwidth: {bandwidth_gb_s:.0f} GB/s")

print(f"\n GPU spec (H100 = 3350 GB/s, H200 = 4800 GB/s)")
print(f"  Achieving >80% of peak is considered good.")

print("\n==== Compute Throughput Measurement ====\n")

print("Measuring GPU compute throughput (TFLOPS):\n")

# Measure FLOPS with matrix multiply (uses Tensor Cores)
for size in [1024, 2048, 4096, 8192]:
    a = torch.randn(size, size, device=DEVICE, dtype=torch.float16)
    b = torch.randn(size, size, device=DEVICE, dtype=torch.float16)

    # Warm up
    for _ in range(50):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    n_iters = max(5, 500 // (size // 1024))
    for _ in range(n_iters):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters

    flops = 2 * size * size * size  # GEMM is 2*M*N*K FLOPs
    tflops = flops / elapsed / 1e12
    print(f"  GEMM {size}×{size}×{size} (FP16): {elapsed*1000:.3f} ms, "
          f"{tflops:.1f} TFLOPS")

print(f"\n  Theoretical peak FP16 Tensor Core:  ~70% of dense peak")
print(f"  (H200/H100 SXM = 989 TFLOPS FP16, A100 = 312 TFLOPS FP16)")

print("\n==== Kernel Launch Overhead ====\n")

print("Measuring the cost of launching a GPU kernel:\n")

# Measure launch overhead with a trivial kernel
x_tiny = torch.randn(1, device=DEVICE)

# Warm up
for _ in range(100):
    _ = x_tiny + x_tiny
torch.cuda.synchronize()

# Measure many tiny launches
n_launches = 10000
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(n_launches):
    _ = x_tiny + x_tiny
torch.cuda.synchronize()
launch_overhead_us = (time.perf_counter() - start) / n_launches * 1e6

print(f"  Kernel launch overhead: ~{launch_overhead_us:.1f} μs per launch")
print(f"  For a model with 1000 kernel launches per forward pass:")
print(f"  → {launch_overhead_us * 1000 / 1000:.1f} ms of pure overhead")
print(f"  This is why CUDA Graphs matter — they reduce this to ~1 launch.")

"""
python -m llm.compilation.06_gpu_hardware

==== GPU HARDWARE — Measuring What the Silicon Does ====

Let's measure actual GPU hardware characteristics to understand the performance constraints kernels operate under.

==== Memory Bandwidth Measurement ====

GPU memory:
Registers       ~256KB per SM    Fastest, per-thread
L1 / Shared     ~128-256KB per SM  Programmer-controlled (shared mem) or automatic (L1)
L2              ~40-80MB total   Shared across all SMs
HBM             ~40-80GB         Main GPU memory, slowest

L1 is per-SM (Streaming Multiprocessor) and is actually split into two parts on NVIDIA GPUs:
- - Automatic L1 cache — caches global memory reads transparently
Shared memory — explicitly managed by the programmer in CUDA/Triton (tl.load into shared). This is the fast scratchpad you control directly.
L2 is shared across all SMs on the chip. Much larger than L1 but slower.

Measuring GPU memory bandwidth (the #1 bottleneck for LLM inference):

     1 MB copy: 0.006 ms, bandwidth: 354 GB/s
    10 MB copy: 0.006 ms, bandwidth: 3480 GB/s
   100 MB copy: 0.053 ms, bandwidth: 3976 GB/s
   500 MB copy: 0.249 ms, bandwidth: 4205 GB/s

 GPU spec (H100 = 3350 GB/s, H200 = 4800 GB/s)
  Achieving >80% of peak is considered good.

==== Compute Throughput Measurement ====

Measuring GPU compute throughput (TFLOPS):

  GEMM 1024×1024×1024 (FP16): 0.012 ms, 184.7 TFLOPS
  GEMM 2048×2048×2048 (FP16): 0.025 ms, 694.2 TFLOPS
  GEMM 4096×4096×4096 (FP16): 0.178 ms, 772.6 TFLOPS
  GEMM 8192×8192×8192 (FP16): 1.642 ms, 669.6 TFLOPS

  Theoretical peak FP16 Tensor Core:  ~70% of dense peak
  (H200/H100 SXM = 989 TFLOPS FP16, A100 = 312 TFLOPS FP16)

==== Kernel Launch Overhead ====

Measuring the cost of launching a GPU kernel:

  Kernel launch overhead: ~6.4 μs per launch
  For a model with 1000 kernel launches per forward pass:
  → 6.4 ms of pure overhead
  This is why CUDA Graphs matter — they reduce this to ~1 launch.
"""