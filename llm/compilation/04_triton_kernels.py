import time
import torch
import torch.nn.functional as F

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
    print(f"Triton version  : {triton.__version__}")
except ImportError:
    HAS_TRITON = False
    raise ImportError("Triton is not installed. Install with: pip install triton")

print("\n==== TRITON — Writing Your Own GPU Kernel ====\n")

print("""
Now we write GPU kernels ourselves using Triton.
This is what Inductor generates automatically, but understanding it by hand is essential for deep comprehension.
""")

print("\n==== First Triton Kernel: Vector Addition ====\n")

print("""
== Understand tl.program_id(axis=0) ==

This maps directly to how GPU grids work. GPUs organize parallel work in a 3D grid — and that's a hardware-level design constraint inherited from CUDA.

axis=0 → X dimension
axis=1 → Y dimension  
axis=2 → Z dimension

1D problem (like vector addition) → only axis=0

# Launch: grid = (num_blocks,)
pid = tl.program_id(axis=0)  # Which block am I along X?

Workers: [pid=0] [pid=1] [pid=2] [pid=3] ...
          ───────────────────────────────────→ axis=0

2D problem (like matrix operations) → axis=0 and axis=1

# Launch: grid = (num_blocks_x, num_blocks_y)
row_id = tl.program_id(axis=0)  # Which block along X?
col_id = tl.program_id(axis=1)  # Which block along Y?
axis=0 (columns)

            0     1     2     3
          ┌─────┬─────┬─────┬─────┐
    0     │(0,0)│(1,0)│(2,0)│(3,0)│
axis=1    ├─────┼─────┼─────┼─────┤
(rows) 1  │(0,1)│(1,1)│(2,1)│(3,1)│
          ├─────┼─────┼─────┼─────┤
    2     │(0,2)│(1,2)│(2,2)│(3,2)│
          └─────┴─────┴─────┴─────┘

3D problem (like batched matrix ops) → axis=0, 1, and 2

# Launch: grid = (num_blocks_x, num_blocks_y, num_blocks_z)
x_id = tl.program_id(axis=0)    # X position
y_id = tl.program_id(axis=1)    # Y position
batch_id = tl.program_id(axis=2) # Which batch?

""")

@triton.jit
def vector_add_kernel(
    a_ptr,        # Pointer to first input vector in GPU memory
    b_ptr,        # Pointer to second input vector in GPU memory
    output_ptr,   # Pointer to output vector in GPU memory
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program handles
):
    """
    Each 'program instance' processes BLOCK_SIZE elements.
    tl.program_id(0) tells us which block we are.

    Think of it like: "I am worker #5, I handle elements 5*1024 to 6*1024"
    """
    # Which block am I?
    pid = tl.program_id(axis=0)

    # Calculate the range of elements this block handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # tl.arange(0, BLOCK_SIZE) creates a vector [0, 1, 2, ..., 1023]

    # After offsets creation, it might look like:
    # pid=0 → offsets = [0, 1, 2, ..., 1023]
    # pid=1 → offsets = [1024, 1025, 1026, ..., 2047]
    # pid=2 → offsets = [2048, 2049, 2050, ..., 3071]

    # Mask to handle the case where n_elements isn't divisible by BLOCK_SIZE
    mask = offsets < n_elements

    # maks is a boolean array for that block
    # assume n_elements=2500 and block size is 1024, any index larger than 2499 is invalid
    # offsets = [2048, 2049, ..., 2499, 2500, 2501, ..., 3071]
    # mask    = [True, True, ..., True, False, False, ..., False]

    # tl.load reads those elements from GPU global memory (DRAM) into registers (ultrafast on-chip storage)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Compute (happens in registers — super fast!)
    output = a + b

    # Store result back to GPU global memory
    tl.store(output_ptr + offsets, output, mask=mask)

# Test it!
def triton_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 1024

    # Calculate grid: how many program instances we need
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the kernel
    vector_add_kernel[grid](a, b, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

# Verify correctness
size = 100_000
a = torch.randn(size, device=DEVICE, dtype=torch.float32)
b = torch.randn(size, device=DEVICE, dtype=torch.float32)

triton_result = triton_vector_add(a, b)
torch_result = a + b

max_diff = (triton_result - torch_result).abs().max().item()
print(f"Vector addition (n={size:,}):")
print(f"  Max difference from PyTorch: {max_diff:.2e}")
print(f"  Correct: {max_diff < 1e-5}")

# Benchmark
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(1000):
    _ = triton_vector_add(a, b)
torch.cuda.synchronize()
triton_us = (time.perf_counter() - start) / 1000 * 1e6

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(1000):
    _ = a + b
torch.cuda.synchronize()
torch_us = (time.perf_counter() - start) / 1000 * 1e6

print(f"\n  Triton kernel: {triton_us:.1f} μs")
print(f"  PyTorch (a+b): {torch_us:.1f} μs")
print(f"  (For simple ops, PyTorch is already well-optimized)")

print("\n==== A More Interesting Kernel: Fused SiLU + Multiply ====\n")

print("""
This kernel demonstrates FUSION — the key optimization.
Instead of: read x → silu(x) → write → read → multiply → write
We do:      read x → silu(x) * gate → write
Saving one full round-trip to GPU memory.

== Understand why [N, 2*D] ==

Assume a layer 
class LLaMAFeedForward(nn.Module):
    def __init__(self, hidden_dim, D):
        self.gate_proj = nn.Linear(hidden_dim, D)  # produces D values
        self.up_proj   = nn.Linear(hidden_dim, D)  # produces D values

    def forward(self, x):
        gate = self.gate_proj(x)   # shape: [N, D]
        up   = self.up_proj(x)     # shape: [N, D]
        return silu(gate) * up     # shape: [N, D]

Input x: [N, hidden_dim]
              │
              ▼
     ┌─── One Big Linear Layer ──-─┐
     │  W_combined: [hidden, 2*D]  │
     └─────────────────────────────┘
              │
              ▼
     combined: [N, 2*D]
     ┌────────────┬────────────┐
     │  first D   │  last D    │
     │   (gate)   │   (up)     │
     └──────┬─────┴─────┬──────┘
            │           │
            ▼           │
        SiLU(gate)      │
            │           │
            ▼           ▼
         silu_gate  ×  up        ← THIS is what the kernel computes
            │
            ▼
     output: [N, D]              ← half the size!
""")

@triton.jit
def fused_silu_mul_kernel(
    x_ptr,          # Input: [N, 2*D] — first half is x, second half is gate
    output_ptr,     # Output: [N, D]
    stride_n,       # Stride along N dimension
    D,              # Half of the last dimension
    N_ELEMENTS,     # Total elements to process
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes: output = silu(x[..., :D]) * x[..., D:]
    This is the SwiGLU activation used in LLaMA/Mistral models.
    """
    pid = tl.program_id(0)
    row = pid # Each program instance handles one entire row
    # i.e.  pid=0 → row 0, pid=1 → row 1, etc.
    # This is different from vector_add where each block handled a chunk — here each block handles a row
    # grid = (N,)  → N program instances, one per row
    # pid=0 → processes row 0: [x0 x1 x2 x3 | g0 g1 g2 g3]
    # pid=1 → processes row 1: [x0 x1 x2 x3 | g0 g1 g2 g3]
    # pid=2 → processes row 2: [x0 x1 x2 x3 | g0 g1 g2 g3]

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    # Load the "gate" part and the "up" part from the SAME row
    # Why stride_n? Tensors in memory are flat. For a [N, 2*D] tensor with D=4:
    # Row 0: [x0 x1 x2 x3 | g0 g1 g2 g3]  ← 8 elements (2*D)
    # Row 1: [x0 x1 x2 x3 | g0 g1 g2 g3]  ← starts at offset 8
    # Row 2: [x0 x1 x2 x3 | g0 g1 g2 g3]  ← starts at offset 16
    # stride_n = 2*D = 8 tells us: jump 8 elements to reach the next row.
    x_vals = tl.load(x_ptr + row * stride_n + col_offsets, mask=mask)
    gate_vals = tl.load(x_ptr + row * stride_n + D + col_offsets, mask=mask)

    # SiLU(x) = x * sigmoid(x)
    silu_x = x_vals * tl.sigmoid(x_vals)

    # Fused multiply
    result = silu_x * gate_vals

    # Store
    tl.store(output_ptr + row * D + col_offsets, result, mask=mask)

def triton_fused_silu_mul(x: torch.Tensor) -> torch.Tensor:
    """x shape: [batch, seq, 2*D] → output shape: [batch, seq, D]"""
    *batch_dims, two_d = x.shape
    D = two_d // 2
    x_flat = x.reshape(-1, two_d) # Collapses all batch dimensions, i.e. [2, 128, 600] → [256, 600]
    N = x_flat.shape[0]
    output = torch.empty(N, D, device=x.device, dtype=x.dtype) # torch.empty allocates memory without initializing (faster than zeros)

    BLOCK_SIZE = triton.next_power_of_2(D) # Powers of 2 are required by tl.arange and enable efficient GPU memory access patterns
    grid = (N,) # launches 256 program instances (one per row)
    fused_silu_mul_kernel[grid](
        x_flat, output, x_flat.stride(0), D, N * D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(*batch_dims, D) # reshape back [256, 300] → [2, 128, 300]

# Compare with PyTorch eager
def pytorch_silu_mul(x: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1] // 2
    return F.silu(x[..., :D]) * x[..., D:]

# test_x = torch.randn(8, 128, 2048, device=DEVICE, dtype=torch.float32)
#   Fused Triton kernel:  0.020 ms
#   Unfused PyTorch:      0.018 ms
test_x = torch.randn(64, 1024, 8192, device=DEVICE, dtype=torch.float32)
triton_out = triton_fused_silu_mul(test_x)
pytorch_out = pytorch_silu_mul(test_x)

max_diff = (triton_out - pytorch_out).abs().max().item()
print(f"Fused SiLU*Mul (shape={list(test_x.shape)}):")
print(f"  Max difference: {max_diff:.2e}")
print(f"  Correct: {max_diff < 1e-2}")  # FP16 tolerance

# Benchmark
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(200):
    _ = triton_fused_silu_mul(test_x)
torch.cuda.synchronize()
fused_ms = (time.perf_counter() - start) / 200 * 1000

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(200):
    _ = pytorch_silu_mul(test_x)
torch.cuda.synchronize()
unfused_ms = (time.perf_counter() - start) / 200 * 1000

print(f"\n  Fused Triton kernel:  {fused_ms:.3f} ms")
print(f"  Unfused PyTorch:      {unfused_ms:.3f} ms")
print(f"  Speedup:              {unfused_ms/fused_ms:.2f}×")
print(f"\n  The speedup comes from eliminating intermediate memory writes.")

print("\n==== Understanding the Grid, Blocks, and Program IDs ====\n")

print("""
When you launch a Triton kernel with kernel[grid](...), here's what happens:

grid = (num_programs,)  or  (grid_x, grid_y)  or  (grid_x, grid_y, grid_z)

Each "program" is like one CUDA thread block.
Inside each program, tl.arange(0, BLOCK_SIZE) creates a vector of indices.
Each element of that vector maps to one thread.

Example: vector of 10,000 elements, BLOCK_SIZE=1024
→ grid = (10,)  → 10 programs
→ Program 0 handles elements [0, 1024)
→ Program 1 handles elements [1024, 2048)
→ ...
→ Program 9 handles elements [9216, 10000)  (with mask for the last 784)

The GPU scheduler distributes these programs across SMs.
If you have 132 SMs and 10 programs, each SM gets ~1 program.
If you have 132 SMs and 10,000 programs, each SM gets ~76 programs.
""")

"""
python -m llm.compilation.04_triton_kernels

Triton version  : 3.5.1

==== TRITON — Writing Your Own GPU Kernel ====

Now we write GPU kernels ourselves using Triton.
This is what Inductor generates automatically, but understanding it by hand is essential for deep comprehension.

==== First Triton Kernel: Vector Addition ====

== Understand tl.program_id(axis=0) ==

This maps directly to how GPU grids work. GPUs organize parallel work in a 3D grid — and that's a hardware-level design constraint inherited from CUDA.

axis=0 → X dimension
axis=1 → Y dimension  
axis=2 → Z dimension

1D problem (like vector addition) → only axis=0

# Launch: grid = (num_blocks,)
pid = tl.program_id(axis=0)  # Which block am I along X?

Workers: [pid=0] [pid=1] [pid=2] [pid=3] ...
          ───────────────────────────────────→ axis=0

2D problem (like matrix operations) → axis=0 and axis=1

# Launch: grid = (num_blocks_x, num_blocks_y)
row_id = tl.program_id(axis=0)  # Which block along X?
col_id = tl.program_id(axis=1)  # Which block along Y?
axis=0 (columns)

            0     1     2     3
          ┌─────┬─────┬─────┬─────┐
    0     │(0,0)│(1,0)│(2,0)│(3,0)│
axis=1    ├─────┼─────┼─────┼─────┤
(rows) 1  │(0,1)│(1,1)│(2,1)│(3,1)│
          ├─────┼─────┼─────┼─────┤
    2     │(0,2)│(1,2)│(2,2)│(3,2)│
          └─────┴─────┴─────┴─────┘

3D problem (like batched matrix ops) → axis=0, 1, and 2

# Launch: grid = (num_blocks_x, num_blocks_y, num_blocks_z)
x_id = tl.program_id(axis=0)    # X position
y_id = tl.program_id(axis=1)    # Y position
batch_id = tl.program_id(axis=2) # Which batch?

Vector addition (n=100,000):
  Max difference from PyTorch: 0.00e+00
  Correct: True

  Triton kernel: 14.7 μs
  PyTorch (a+b): 6.1 μs
  (For simple ops, PyTorch is already well-optimized)

==== A More Interesting Kernel: Fused SiLU + Multiply ====

This kernel demonstrates FUSION — the key optimization.
Instead of: read x → silu(x) → write → read → multiply → write
We do:      read x → silu(x) * gate → write
Saving one full round-trip to GPU memory.

== Understand why [N, 2*D] ==

Assume a layer 
class LLaMAFeedForward(nn.Module):
    def __init__(self, hidden_dim, D):
        self.gate_proj = nn.Linear(hidden_dim, D)  # produces D values
        self.up_proj   = nn.Linear(hidden_dim, D)  # produces D values

    def forward(self, x):
        gate = self.gate_proj(x)   # shape: [N, D]
        up   = self.up_proj(x)     # shape: [N, D]
        return silu(gate) * up     # shape: [N, D]

Input x: [N, hidden_dim]
              │
              ▼
     ┌─── One Big Linear Layer ──-─┐
     │  W_combined: [hidden, 2*D]  │
     └─────────────────────────────┘
              │
              ▼
     combined: [N, 2*D]
     ┌────────────┬────────────┐
     │  first D   │  last D    │
     │   (gate)   │   (up)     │
     └──────┬─────┴─────┬──────┘
            │           │
            ▼           │
        SiLU(gate)      │
            │           │
            ▼           ▼
         silu_gate  ×  up        ← THIS is what the kernel computes
            │
            ▼
     output: [N, D]              ← half the size!

Fused SiLU*Mul (shape=[8, 128, 2048]):
  Max difference: 9.54e-07
  Correct: True

  Fused Triton kernel:  0.020 ms
  Unfused PyTorch:      0.018 ms
  Speedup:              0.91×    

For a shape of (8, 128, 2048) that's only ~2M elements — the GPU finishes it so fast that kernel launch 
overhead dominates, and your custom Triton kernel has more overhead than PyTorch's built-in.

Fused SiLU*Mul (shape=[64, 1024, 8192]):
  Max difference: 2.86e-06
  Correct: True

  Fused Triton kernel:  0.746 ms
  Unfused PyTorch:      1.710 ms
  Speedup:              2.29×

The speedup comes from eliminating intermediate memory writes.

==== Understanding the Grid, Blocks, and Program IDs ====


When you launch a Triton kernel with kernel[grid](...), here's what happens:

grid = (num_programs,)  or  (grid_x, grid_y)  or  (grid_x, grid_y, grid_z)

Each "program" is like one CUDA thread block.
Inside each program, tl.arange(0, BLOCK_SIZE) creates a vector of indices.
Each element of that vector maps to one thread.

Example: vector of 10,000 elements, BLOCK_SIZE=1024
→ grid = (10,)  → 10 programs
→ Program 0 handles elements [0, 1024)
→ Program 1 handles elements [1024, 2048)
→ ...
→ Program 9 handles elements [9216, 10000)  (with mask for the last 784)

The GPU scheduler distributes these programs across SMs.
If you have 132 SMs and 10 programs, each SM gets ~1 program.
If you have 132 SMs and 10,000 programs, each SM gets ~76 programs.
"""