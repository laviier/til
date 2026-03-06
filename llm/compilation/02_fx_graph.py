import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("""
The FX Graph is a DAG (directed acyclic graph) of operations.
Each node represents one operation. Let's inspect it in detail.

Note: FX stands for Functional Transformations. It's PyTorch's IR (intermediate representation) for capturing and transforming neural network computation as a graph. The name reflects its core purpose — representing the model as a pure functional graph that can be analyzed, transformed, and optimized.
""")

print("\n==== Building an FX Graph Manually ====\n")

# Trace our model to get an FX graph
from torch.fx import symbolic_trace

# Use a simpler model for cleaner tracing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        y = self.linear(x)
        y = F.relu(y)
        y = y + x  # residual
        return y

simple_model = SimpleModel().to(DEVICE).half()
traced = symbolic_trace(simple_model)

print("FX Graph nodes:")
print(f"{'Op':<15} {'Name':<20} {'Target':<40} {'Args'}")
print("-" * 95)
for node in traced.graph.nodes:
    target_str = str(node.target)[:38]
    args_str = str(node.args)[:30]
    print(f"{node.op:<15} {node.name:<20} {target_str:<40} {args_str}")

print(f"\nTotal nodes: {len(list(traced.graph.nodes))}")
print("\nNode types:")
print("  placeholder  = input to the graph")
print("  call_module  = calling a sub-module (e.g., self.linear)")
print("  call_function = calling a function (e.g., torch.relu)")
print("  output       = the return value")

print("\n==== Graph Visualization as Python Code ====\n")

print("The FX graph can be printed as readable Python:")
print(traced.code)

print("==== Graph Transformations (What Inductor Passes Do) ====")

print("""
Before Inductor generates kernels, it transforms the graph.
Let's do a simple transformation: fuse relu + add into one operation.
""")

from torch.fx import Graph, Node

def fuse_relu_add_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Example pass: find relu followed by add and annotate them."""
    graph = gm.graph
    fusions_found = 0
    for node in graph.nodes:
        if node.op == "call_function" and node.target == torch.relu:
            # Check if the output of relu feeds into an add
            for user in node.users:
                if user.op == "call_function" and user.target == torch.add:
                    fusions_found += 1
                    print(f"  Found fusible pattern: {node.name} → {user.name}")
    print(f"  Total fusible patterns: {fusions_found}")
    return gm

# Trace a model with relu + add pattern
class FusibleModel(nn.Module):
    def forward(self, x, y):
        a = torch.relu(x)
        b = torch.add(a, y)
        return b

fusible = symbolic_trace(FusibleModel())
print("Scanning graph for fusible patterns:")
fuse_relu_add_pass(fusible)
print("\nIn real Inductor, this fusion would combine both ops into one kernel.")

"""
python -m llm.compilation.02_fx_graph

The FX Graph is a DAG (directed acyclic graph) of operations.
Each node represents one operation. Let's inspect it in detail.

Note: FX stands for Functional Transformations. It's PyTorch's IR (intermediate representation) for capturing and transforming neural network computation as a graph. The name reflects its core purpose — representing the model as a pure functional graph that can be analyzed, transformed, and optimized.

==== Building an FX Graph Manually ====

FX Graph nodes:
Op              Name                 Target                                   Args
-----------------------------------------------------------------------------------------------
placeholder     x                    x                                        ()
call_module     linear               linear                                   (x,)
call_function   relu                 <function relu at 0x7841ebd3d9e0>        (linear,)
call_function   add                  <built-in function add>                  (relu, x)
output          output               output                                   (add,)

Total nodes: 5

Node types:
  placeholder  = input to the graph
  call_module  = calling a sub-module (e.g., self.linear)
  call_function = calling a function (e.g., torch.relu)
  output       = the return value

==== Graph Visualization as Python Code ====

The FX graph can be printed as readable Python:

def forward(self, x):
    linear = self.linear(x)
    relu = torch.nn.functional.relu(linear, inplace = False);  linear = None
    add = relu + x;  relu = x = None
    return add
    
==== Graph Transformations (What Inductor Passes Do) ====

Before Inductor generates kernels, it transforms the graph.
Let's do a simple transformation: fuse relu + add into one operation.

Scanning graph for fusible patterns:
  Found fusible pattern: relu → add
  Total fusible patterns: 1

In real Inductor, this fusion would combine both ops into one kernel.
"""