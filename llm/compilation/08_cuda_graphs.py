import time
import torch

from llm.mini_transformer import MiniFfnBlock

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("\n==== CUDA GRAPHS — Eliminating Launch Overhead ====\n")

print("""
CUDA Graphs record a sequence of kernel launches and replay them with a single CPU command. Let's see the difference.
""")

print("\n==== Manual CUDA Graph Capture and Replay ====\n")

# Create a simple multi-kernel workload
graph_model = MiniFfnBlock(512).to(DEVICE).half()
graph_input = torch.randn(4, 32, 512, device=DEVICE, dtype=torch.float16)

# Warm up
for _ in range(30):
    _ = graph_model(graph_input)
torch.cuda.synchronize()

# --- Without CUDA Graph ---
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(500):
    _ = graph_model(graph_input)
torch.cuda.synchronize()
no_graph_ms = (time.perf_counter() - start) / 500 * 1000

# --- With CUDA Graph ---
# Step 1: Allocate static buffers (CUDA graphs need fixed memory addresses)
static_input = graph_input.clone()

# Step 2: Warm up (required before capture)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(30):
        static_output = graph_model(static_input)
torch.cuda.current_stream().wait_stream(s)

# Step 3: Capture the graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): # CUDA driver is in "recording mode"
    static_output = graph_model(static_input)

# Step 4: Replay!
# Copy new data into the static input buffer
static_input.copy_(graph_input) # Put new data at the SAME address

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(500):
    g.replay()  # Single CPU command replays ALL kernel launches
torch.cuda.synchronize()
graph_ms = (time.perf_counter() - start) / 500 * 1000

# Verify correctness
eager_output = graph_model(graph_input)
graph_output = static_output
max_diff = (eager_output - graph_output).abs().max().item()

print(f"Without CUDA Graph: {no_graph_ms:.3f} ms")
print(f"With CUDA Graph:    {graph_ms:.3f} ms")
print(f"Speedup:            {no_graph_ms/graph_ms:.2f}×")
print(f"Max difference:     {max_diff:.2e} (should be ~0)")
print(f"\nThe speedup comes from eliminating per-kernel CPU launch overhead.")

print("\n==== How vLLM Uses CUDA Graphs (Conceptual) ====\n")

print("""
vLLM's piecewise CUDA graph approach:

  1. The model is split at attention boundaries:
     [Piece 0: norm+proj] → [Attention] → [Piece 1: proj+MLP] → [Attention] → ...

  2. Each piece (excluding attention) is captured as a CUDA graph.

  3. At runtime:
     - Replay CUDA Graph 0 (norm + QKV projection)
     - Run attention eagerly (flexible, handles variable sequence lengths)
     - Replay CUDA Graph 1 (output projection + MLP)
     - Run attention eagerly
     - ...

  4. For decode (batch of single tokens), vLLM can also capture a FULL
     CUDA graph that includes attention, since decode attention is simpler.

  This gives the best of both worlds:
  - Fast execution (CUDA graphs) for the "easy" parts
  - Flexibility (eager) for the "hard" parts (attention)
""")

"""
python -m llm.compilation.08_cuda_graphs

==== CUDA GRAPHS — Eliminating Launch Overhead ====

CUDA Graphs record a sequence of kernel launches and replay them with a single CPU command. Let's see the difference.

  COMPILATION (happens before capture):                         
  ┌──────────────────────────────────────                   
  │ Triton @jit kernel → compiled to PTX → GPU machine code    
  │ cuBLAS selects best pre-compiled kernel for this size      
  │ PyTorch compiles any torch.compile'd functions             
  └──────────────────────────────────────                     
This produces KERNEL BINARIES sitting on the GPU           

  CUDA GRAPH (happens after compilation):                       
  ┌──────────────────────────────────────                      
  │ Records: "Run kernel A, then B, then C"                    
  │ With: these exact arguments, memory addresses              
  │ The kernels are ALREADY COMPILED                           
  │ The graph just records the ORDER and ARGUMENTS             
  └──────────────────────────────────────                     
                                                                
  COMPILATION = creating the recipe (done once)                 
  CUDA GRAPH  = pre-ordering 500 copies of the meal            
│               (no need to re-explain the recipe each time)  

==== Manual CUDA Graph Capture and Replay ====

Without CUDA Graph: 0.076 ms
With CUDA Graph:    0.017 ms
Speedup:            4.49×
Max difference:     0.00e+00 (should be ~0)

The speedup comes from eliminating per-kernel CPU launch overhead.
"""