import time
import torch

from llm.mini_transformer import MiniTransformer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

mini_llm = MiniTransformer(vocab_size=32000, hidden_size=512, num_layers=4).to(DEVICE).half()
test_input = torch.randint(0, 32000, (4, 64), device=DEVICE)  # (Batch, Toekn) integer token IDs
# generate random integers between 0 and 31999 (the vocab token IDs) and the shape is 4 sequences, each 64 tokens long

print("\n==== Benchmark: Eager vs Compiled vs CUDA Graph ====\n")

print("""
Eager:                [k1] [k2] [k3] [k4] [k5] [k6]   ← 6 kernels, 6 CPU launches
Manual CUDA Graph:    [k1→k2→k3→k4→k5→k6]             ← 6 kernels, 1 GPU replay
Compiled (Inductor):  [K1] [K2] [K3]                  ← 3 fused kernels, 3 CPU launches
Compiled + CUDAGraph: [K1→K2→K3]                      ← 3 fused kernels, 1 GPU replay
""")

results = {}

# --- Eager ---
for _ in range(50):
    _ = mini_llm(test_input)
torch.cuda.synchronize()

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(200):
    _ = mini_llm(test_input)
torch.cuda.synchronize()
results["Eager"] = (time.perf_counter() - start) / 200 * 1000

# --- torch.compile (Inductor) ---
compiled_llm = torch.compile(mini_llm, backend="inductor")
# Warm up (triggers compilation)
for _ in range(50):
    _ = compiled_llm(test_input)
torch.cuda.synchronize()

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(200):
    _ = compiled_llm(test_input)
torch.cuda.synchronize()
results["Compiled (Inductor)"] = (time.perf_counter() - start) / 200 * 1000

# --- torch.compile with reduce-overhead (includes CUDA graphs) ---
compiled_llm_fast = torch.compile(mini_llm, backend="inductor", mode="reduce-overhead")
for _ in range(50):
    _ = compiled_llm_fast(test_input)
torch.cuda.synchronize()

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(200):
    _ = compiled_llm_fast(test_input)
torch.cuda.synchronize()
results["Compiled + CUDAGraph"] = (time.perf_counter() - start) / 200 * 1000

# --- Manual CUDA Graph ---
static_in = test_input.clone()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(30):
        _ = mini_llm(static_in)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = mini_llm(static_in)

static_in.copy_(test_input)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(200):
    g.replay()
torch.cuda.synchronize()
results["Manual CUDA Graph"] = (time.perf_counter() - start) / 200 * 1000

# Print results
print(f"{'Method':<30} {'Time (ms)':>10} {'Speedup':>10}")
print("-" * 52)
baseline = results["Eager"]
for name, ms in results.items():
    speedup = baseline / ms
    bar = "█" * int(speedup * 10)
    print(f"{name:<30} {ms:>9.3f}  {speedup:>8.2f}×  {bar}")

print("\n==== Memory Bandwidth Analysis ====\n")

print("""
Why does fusion help? Let's calculate the memory traffic.

For one MiniFfnBlock (hidden_size=512, batch=4, seq=64):
  Input tensor:          4 × 64 × 512  × 2 bytes (FP16) = 256 KB
  Intermediate tensor:   4 × 64 × 2048 × 2 bytes (FP16) = 1 MB
  up_proj weights:       512  × 2048   × 2 bytes (FP16) = 2 MB
  down_proj weights:     2048 × 512    × 2 bytes (FP16) = 2 MB

  EAGER MODE (each op reads/writes tensors to HBM separately):
    LayerNorm:   read 256KB  + write 256KB               = 512 KB
    up_proj:     read 256KB  + read 2MB weights + write 1MB = 3.25 MB
    SiLU:        read 1MB    + write 1MB                 = 2 MB
    down_proj:   read 1MB    + read 2MB weights + write 256KB = 3.25 MB
    Residual:    read 256KB  + read 256KB + write 256KB  = 768 KB
    Total:       ~9.8 MB of memory traffic per FFN block

  COMPILED (Inductor fuses LayerNorm+up_proj and SiLU+down_proj):
    Fused norm+up_proj:   read 256KB + read 2MB weights + write 1MB = 3.25 MB
    Fused silu+down_proj: read 1MB   + read 2MB weights + write 256KB = 3.25 MB
    Residual:             read 256KB + read 256KB + write 256KB = 768 KB
    Total:                ~7.3 MB of memory traffic per FFN block

  Reduction: ~25% less memory traffic → speedup for bandwidth-bound ops.
  (Weight reads dominate and can't be fused away — that's the hard floor.)
""")

"""
python -m llm.compilation.09_full_pipeline

==== Benchmark: Eager vs Compiled vs CUDA Graph ====

Eager:                [k1] [k2] [k3] [k4] [k5] [k6]   ← 6 kernels, 6 CPU launches
Manual CUDA Graph:    [k1→k2→k3→k4→k5→k6]             ← 6 kernels, 1 GPU replay
Compiled (Inductor):  [K1] [K2] [K3]                  ← 3 fused kernels, 3 CPU launches
Compiled + CUDAGraph: [K1→K2→K3]                      ← 3 fused kernels, 1 GPU replay

Method                          Time (ms)    Speedup
----------------------------------------------------
Eager                              1.741      1.00×  ██████████
Compiled (Inductor)                1.167      1.49×  ██████████████
Compiled + CUDAGraph               0.292      5.96×  ███████████████████████████████████████████████████████████
Manual CUDA Graph                  0.316      5.50×  ███████████████████████████████████████████████████████
"""