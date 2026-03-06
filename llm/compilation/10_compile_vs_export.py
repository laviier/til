"""
torch.compile vs torch.export — What's the difference?

| Feature            | torch.compile                                      | torch.export                                          |
|--------------------|---------------------------------------------------|-------------------------------------------------------|
| Compilation Model  | Just-In-Time (JIT)                                | Ahead-Of-Time (AOT)                                   |
| Graph Capture      | Partial; graph breaks fall back to Python eager   | Full graph required; errors on untraceable code       |
| Primary Goal       | Speed up existing code with minimal changes       | Portable graph for deployment outside Python runtime  |
| Flexibility        | High; works on arbitrary Python code              | Low; may need torch._check() / torch.cond() rewrites  |
| Output Artifact    | Optimized kernels in the existing Python process  | ExportedProgram (.pt2) usable in C++ / AOTInductor    |
"""

import torch
from llm.mini_transformer import MiniFfnBlock

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model = MiniFfnBlock(1024).to(DEVICE).half()
x = torch.randn(2, 64, 1024, device=DEVICE, dtype=torch.float16)

print("==== torch.compile (JIT) ====\n")

# torch.compile traces lazily on first call.
# Graph breaks are allowed — untraceable code falls back to Python eager.
compiled = torch.compile(model, backend="inductor")
out = compiled(x)
print(f"Output shape: {out.shape}")
print("Compiled successfully — graph breaks are silently handled.\n")


print("==== torch.export (AOT) ====\n")

# torch.export requires a FULL, clean graph.
# It will error if there are graph breaks or data-dependent control flow.
try:
    exported = torch.export.export(model, (x,))
    print("Exported graph:")
    print(exported.graph_module.graph)

    # The ExportedProgram can be saved to disk and loaded in non-Python runtimes
    # torch.export.save(exported, "model.pt2")
    # loaded = torch.export.load("model.pt2")
    print("\nExportedProgram can be serialized with torch.export.save(ep, 'model.pt2')")
    print("and loaded in C++ via AOTInductor without a Python runtime.\n")
except Exception as e:
    print(f"Export failed: {e}\n")


print("==== Graph Break Demo ====\n")

# A model with a Python print() causes a graph break in torch.compile
# but will FAIL torch.export entirely.
class ModelWithBreak(torch.nn.Module):
    def forward(self, x):
        x = x * 2
        if x.sum() > 0:   # data-dependent control flow — graph break!
            x = x + 1
        return x

model_break = ModelWithBreak().to(DEVICE)
x_small = torch.randn(4, device=DEVICE)

# torch.compile handles it gracefully with graph breaks
compiled_break = torch.compile(model_break)
out = compiled_break(x_small)
print(f"torch.compile with graph break: OK, output shape {out.shape}")

# torch.export will fail on the same model
try:
    torch.export.export(model_break, (x_small,))
    print("torch.export: OK (unexpected)")
except Exception as e:
    print(f"torch.export with graph break: FAILED as expected")
    print(f"  Reason: {type(e).__name__}")
