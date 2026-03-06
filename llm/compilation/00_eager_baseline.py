import time
import torch

from llm.mini_transformer import MiniFfnBlock

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model_eager = MiniFfnBlock(1024).to(DEVICE).half()
dummy_input = torch.randn(32, 512, 1024, device=DEVICE, dtype=torch.float16)

# Check weights exist
print("Check weights")
print(model_eager.up_proj.weight.shape)    # torch.Size([4096, 1024])
print(model_eager.down_proj.weight.shape)   # torch.Size([1024, 4096])
print(model_eager.norm.weight.shape)        # torch.Size([1024])

# Warm up
print("Starting warm up")
for _ in range(3):
    print(model_eager(dummy_input))
print("Warm up completed")
torch.cuda.synchronize()

# Benchmark eager
start = time.perf_counter()
for _ in range(100):
    model_eager(dummy_input)
torch.cuda.synchronize()
eager_time = (time.perf_counter() - start) / 100 * 1000

print(f"Eager mode: {eager_time:.3f} ms per forward pass")
print(f"  → Each of the ~6 operations is a separate kernel launch")
print(f"  → Data is read/written to GPU memory between every operation")

"""
python -m llm.compilation.00_eager_baseline

Check weights
torch.Size([4096, 1024])
torch.Size([1024, 4096])
torch.Size([1024])
Starting warm up
tensor([[[-1.0215,  1.1875,  0.5635,  ...,  0.9409, -1.0742, -1.1924],
         ...,
         [-1.1455, -1.1221,  1.4746,  ...,  0.6821, -1.0039, -0.9229]],

        [[-0.6812,  1.4814, -0.4656,  ..., -0.7275, -2.0645, -1.0850],
         ...,
         [-0.2271,  1.0801,  0.6553,  ..., -0.2834, -1.7188, -0.4119]],

        [[ 1.0986, -0.6719,  0.3428,  ..., -0.7422,  0.3623, -0.6318],
         ...,
         [-1.0938,  1.2998,  0.7949,  ..., -0.3577, -0.5957,  0.5771]],

        ...,

        [[ 0.0282, -0.1921,  2.7441,  ..., -2.1191, -0.0673, -2.2520],
         ...,
         [-0.2791,  1.9277, -0.7749,  ...,  1.2236,  0.4026,  1.6045]],

        [[-0.7656,  0.5068, -0.5913,  ...,  0.3157, -0.4268,  0.2861],
         ...,
         [ 1.5127, -1.7275,  1.1621,  ..., -0.8667,  1.5127,  0.7085]],

        [[ 0.1360, -0.7920,  0.2217,  ...,  0.3525,  0.0681,  2.0391],
         ...,
         [-0.0425, -1.1426, -0.8145,  ..., -0.7314,  1.9336, -0.4072]]],
       device='cuda:0', dtype=torch.float16, grad_fn=<AddBackward0>)
...
Warm up completed
Eager mode: 0.511 ms per forward pass
  → Each of the ~6 operations is a separate kernel launch
  → Data is read/written to GPU memory between every operation
"""
