import time
import torch 
import triton
import triton.language as tl

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("\n==== cuBLAS vs CUTLASS vs TRITON — The Kernel Zoo ====\n")

print("""
Different kernel libraries excel at different things. Let's benchmark them head-to-head on matrix multiplication.
""")

print("\n==== cuBLAS (via torch.mm) — The Default GEMM ====\n")

print("cuBLAS is NVIDIA's closed-source, hand-tuned BLAS library.")
print("torch.mm() dispatches to cuBLAS automatically.\n")

M, N, K = 4096, 4096, 4096
a_fp16 = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
b_fp16 = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

# Warm up
for _ in range(50):
    _ = torch.mm(a_fp16, b_fp16)
torch.cuda.synchronize()

# Benchmark cuBLAS
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(500):
    _ = torch.mm(a_fp16, b_fp16)
torch.cuda.synchronize()
cublas_ms = (time.perf_counter() - start) / 500 * 1000
cublas_tflops = 2 * M * N * K / (cublas_ms / 1000) / 1e12

print(f"  cuBLAS GEMM ({M}×{K} × {K}×{N}, FP16):")
print(f"    Time:   {cublas_ms:.3f} ms")
print(f"    TFLOPS: {cublas_tflops:.1f}")

print("\n==== Triton Matrix Multiply ====\n")

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Simple Triton matmul — not fully optimized but demonstrates the concept."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the block of C this program is responsible for
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first block of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_K):
        # Load blocks of A and B
        a_block = tl.load(a_ptrs, mask=(offs_m[:, None] < M) &
                            (offs_k[None, :] + k_start < K), other=0.0)
        b_block = tl.load(b_ptrs, mask=(offs_k[:, None] + k_start < K) &
                            (offs_n[None, :] < N), other=0.0)

        # Matrix multiply-accumulate
        acc += tl.dot(a_block, b_block)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)

def triton_matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c

# Verify correctness
c_triton = triton_matmul(a_fp16, b_fp16)
c_cublas = torch.mm(a_fp16, b_fp16)
max_diff = (c_triton.float() - c_cublas.float()).abs().max().item()
print(f"  Max difference from cuBLAS: {max_diff:.4f}")
print(f"  (FP16 matmul has inherent numerical differences)")

# Benchmark Triton matmul
for _ in range(50):
    _ = triton_matmul(a_fp16, b_fp16)
torch.cuda.synchronize()

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(500):
    _ = triton_matmul(a_fp16, b_fp16)
torch.cuda.synchronize()
triton_mm_ms = (time.perf_counter() - start) / 500 * 1000
triton_tflops = 2 * M * N * K / (triton_mm_ms / 1000) / 1e12

print(f"\n  Triton GEMM (simple, not auto-tuned):")
print(f"    Time:   {triton_mm_ms:.3f} ms")
print(f"    TFLOPS: {triton_tflops:.1f}")
print(f"\n  cuBLAS vs Triton: cuBLAS is {triton_mm_ms/cublas_ms:.1f}× faster")
print(f"  (cuBLAS has years of hand-tuning; Triton can close the gap with auto-tune)")

print("""
""")

print("\n==== When to Use Which Kernel Library ====\n")

print("""
Decision tree for kernel selection:

  Is it a standard dense matmul (FP16/BF16/FP32)?
  ├── YES → Use cuBLAS (torch.mm). It's the fastest for standard types.
  └── NO
      │
      Is it a quantized matmul (FP8, INT8, INT4)?
      ├── YES → Use CUTLASS (custom epilogue for dequantization)
      │         or specialized kernels (Marlin, Machete for INT4)
      └── NO
          │
          Is it an element-wise or reduction operation?
          ├── YES → Let Inductor generate a fused Triton kernel
          │         (or write a custom Triton kernel for special cases)
          └── NO
              │
              Is it attention?
              ├── YES → Use FlashAttention / FlashInfer / Triton attention
              └── NO → Write a custom Triton or CUDA kernel

  Key insight: cuBLAS for standard GEMM, CUTLASS for quantized GEMM,
  Triton for everything else (fusions, attention, custom ops).

┌───────────┬────────────────┬────────────────┬──────────────────┐
│           │  cuBLAS        │  CUTLASS       │  Triton          │
├───────────┼────────────────┼────────────────┼──────────────────┤
│ Written   │ CUDA C/C++     │ CUDA C++       │ Python           │
│ in        │ (closed)       │ (open)         │ (open)           │
│           │                │                │                  │
│ You call  │ C/C++ API      │ C++ templates  │ Python with      │
│ it from   │ OR Python      │ (mostly C++)   │ @triton.jit      │
│           │ (via PyTorch)  │                │                  │
│           │                │                │                  │
│ You       │ Nothing        │ Swap/configure │ Write full       │
│ customize │ (fixed menu)   │ components     │ algorithm        │
│           │                │ (LEGO blocks)  │ (blank canvas)   │
│           │                │                │                  │
│ Scope     │ Standard BLAS  │ GEMM variants  │ Anything on GPU  │
└───────────┴────────────────┴────────────────┴──────────────────┘

The Historical Timeline
2007 │ CUDA released
     │ People write raw CUDA kernels in C/C++
     │
2010 │ cuBLAS released  
     │ "Stop writing matmul yourself, use our optimized library"
     │ Called from C/C++ only
     │
2014 │ cuDNN released (for neural network operations)
     │
2016 │ PyTorch released
     │ PyTorch wraps cuBLAS internally
     │ Now you can call cuBLAS from Python without knowing it
     │ torch.mm() → calls cuBLAS under the hood
     │
2017 │ CUTLASS released (open-source GEMM templates)
     │ "If cuBLAS doesn't support your use case, build your own"
     │
2021 │ Triton released by OpenAI
     │ "Write GPU kernels in Python, not C++"
     │
     ▼

""")


"""
python -m llm.compilation.07_kernel_zoo

==== cuBLAS vs CUTLASS vs TRITON — The Kernel Zoo ====

Different kernel libraries excel at different things. Let's benchmark them head-to-head on matrix multiplication.

==== cuBLAS (via torch.mm) — The Default GEMM ====

cuBLAS is NVIDIA's closed-source, hand-tuned BLAS library.
torch.mm() dispatches to cuBLAS automatically.

  cuBLAS GEMM (4096×4096 × 4096×4096, FP16):
    Time:   0.189 ms
    TFLOPS: 726.6

==== Triton Matrix Multiply ====

  Max difference from cuBLAS: 0.0000
  (FP16 matmul has inherent numerical differences)

  Triton GEMM (simple, not auto-tuned):
    Time:   0.425 ms
    TFLOPS: 323.3

  cuBLAS vs Triton: cuBLAS is 2.2× faster
  (cuBLAS has years of hand-tuning; Triton can close the gap with auto-tune)
"""