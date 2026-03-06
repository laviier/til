import torch
import torch.nn as nn

from llm.mini_transformer import MiniFfnBlock

print("""
Dynamo intercepts Python bytecode to capture what the model does as a graph. 

We use torch._dynamo.export() to get the FX graph without compiling it. This shows what Dynamo "sees".
""")

print("==== Exporting the FX Graph ====")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model_eager = MiniFfnBlock(1024).to(DEVICE).half()
# Use a smaller input for cleaner graph output
small_input = torch.randn(2, 4, 1024, device=DEVICE, dtype=torch.float16)

# Export the graph (Dynamo traces without compiling)
try:
    # https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html
    exported = torch.export.export(
        model_eager,
        (small_input,),
    )
    print("Exported graph structure:")
    print(exported.graph_module.graph)
    print("Exported graph guards:")
    for g in exported.guards: # exported.guards is a torch._guards.GuardsSet object
        # g.name — the tensor/variable being guarded
        # g.guard_types — what kind of guard (e.g. TENSOR_MATCH, TYPE_MATCH)
        # g.code_list — the actual condition as a list of code strings
        print(f"  name={g.name}, type={g.guard_types}, code={g.code_list}")
except Exception as e:
    # Fallback for older PyTorch versions
    print(f"(torch.export not available in this PyTorch version: {e})")
    print("Using torch._dynamo.export instead...")
    try:
        from torch._dynamo import export
        gm, guards = export(model_eager, small_input)
        print("Exported graph structure:")
        print(gm.graph)
        print("Exported graph guards:")
        for g in guards:
            print(f"  {g}")
    except Exception as e2:
        print(f"Export also failed: {e2}")
        print("Skipping graph export visualization.")

print("==== Understanding Guards ====")

print("""
Guards are conditions Dynamo records about the inputs.
If a guard fails on a future call, Dynamo recompiles.

Common guards:
  - tensor.shape[0] == 2        (batch size)
  - tensor.dtype == float16      (data type)
  - tensor.device == cuda:0      (device)
  - tensor.requires_grad == False

In vLLM, guards on batch size are DROPPED (using symbolic shapes) so that one compiled graph works for all batch sizes.
""")

print("==== Graph Breaks Demo ====")

print("""
A 'graph break' happens when Dynamo encounters something it can't trace.
""")

class ModelWithGraphBreak(nn.Module):
    def forward(self, x):
        x = x * 2
        print("This print causes a graph break!")  # <-- graph break!
        x = x + 1
        return x


# Count compilations
compile_count = 0
original_compile_count = compile_count


def counting_backend(gm, example_inputs):
    print("Exported graph structure:")
    print(gm.graph)

    global compile_count
    compile_count += 1
    print(f"  [Backend called] Graph #{compile_count} with "
          f"{len(list(gm.graph.nodes))} nodes")
    # Print the graph
    for node in gm.graph.nodes:
        print(f"    {node.op:15s} | {node.name}")
    return gm  # Return unmodified (eager execution)


model_break = ModelWithGraphBreak()
compiled_break = torch.compile(model_break, backend=counting_backend)

print("Calling model with graph break:")
compile_count = 0
_ = compiled_break(torch.randn(4, device=DEVICE))

print(f"\nDynamo created {compile_count} separate graph(s) due to the print().")
print("Without the print(), it would be 1 graph.")


"""
python -m llm.compilation.01_dynamo_graph_capture.py

==== Exporting the FX Graph ====
Exported graph structure:
graph():
    %p_norm_weight : [num_users=1] = placeholder[target=p_norm_weight]
    %p_norm_bias : [num_users=1] = placeholder[target=p_norm_bias]
    %p_up_proj_weight : [num_users=1] = placeholder[target=p_up_proj_weight]
    %p_down_proj_weight : [num_users=1] = placeholder[target=p_down_proj_weight]
    %x : [num_users=2] = placeholder[target=x]
    %layer_norm : [num_users=1] = call_function[target=torch.ops.aten.layer_norm.default](args = (%x, [1024], %p_norm_weight, %p_norm_bias), kwargs = {})
    %linear : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%layer_norm, %p_up_proj_weight), kwargs = {})
    %silu : [num_users=1] = call_function[target=torch.ops.aten.silu.default](args = (%linear,), kwargs = {})
    %linear_1 : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%silu, %p_down_proj_weight), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%linear_1, %x), kwargs = {})
    return (add,)
Exported graph guards:
(torch.export not available in this PyTorch version: 'ExportedProgram' object has no attribute 'guards')
Using torch._dynamo.export instead...
/home/ubuntu/github/til/llm/compilation/01_dynamo_graph_capture.py:34: FutureWarning: export(f, *args, **kwargs) is deprecated, use export(f)(*args, **kwargs) instead.  If you don't migrate, we may break your export call in the future if your user defined kwargs conflict with future kwargs added to export(f).
  gm, guards = export(model_eager, small_input)
Exported graph structure:
graph():
    %l_x_ : [num_users=2] = placeholder[target=arg0]
    %x : [num_users=1] = call_module[target=L__self___norm](args = (%l_x_,), kwargs = {})
    %x_1 : [num_users=1] = call_module[target=L__self___up_proj](args = (%x,), kwargs = {})
    %x_2 : [num_users=1] = call_function[target=torch.nn.functional.silu](args = (%x_1,), kwargs = {})
    %x_3 : [num_users=1] = call_module[target=L__self___down_proj](args = (%x_2,), kwargs = {})
    %x_4 : [num_users=1] = call_function[target=operator.add](args = (%x_3, %l_x_), kwargs = {})
    return [x_4]
Exported graph guards:
  Name: ''
    Source: global
    Create Function: AUTOGRAD_SAVED_TENSORS_HOOKS
    Guard Types: ['AUTOGRAD_SAVED_TENSORS_HOOKS']
    Code List: ['torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None']
    Object Weakref: None
    Guarded Class Weakref: None

  Name: "L['x']"
    Source: local
    Create Function: TENSOR_MATCH
    Guard Types: ['TYPE_MATCH', 'TENSOR_MATCH']
    Code List: ["___check_type_id(L['x'], 586531840)", "str(L['x'].dtype) == 'torch.float16'", "str(L['x'].device) == 'cuda:0'", "L['x'].requires_grad == False", "L['x'].ndimension() == 3", "hasattr(L['x'], '_dynamo_dynamic_indices') == False"]
    Object Weakref: <weakref at 0x7a9582d338d0; to 'Tensor' at 0x7a993df42d50>
    Guarded Class Weakref: <weakref at 0x7a98d05d7880; to 'torch._C._TensorMeta' at 0x22f5c400 (Tensor)>

  Name: "L['self'].norm"
    Source: local_specialized_nn_module
    Create Function: NN_MODULE
    Guard Types: ['ID_MATCH']
    Code List: ["___check_obj_id(L['self'].norm, 134798827026960)"]
    Object Weakref: <weakref at 0x7a9582a48c70; to 'LayerNorm' at 0x7a994c337a10>
    Guarded Class Weakref: <weakref at 0x7a983a3601d0; to 'type' at 0x23702060 (LayerNorm)>

  Name: ''
    Source: global
    Create Function: DEFAULT_DEVICE
    Guard Types: ['DEFAULT_DEVICE']
    Code List: ['utils_device.CURRENT_DEVICE == None']
    Object Weakref: None
    Guarded Class Weakref: None

  Name: ''
    Source: global
    Create Function: GRAD_MODE
    Guard Types: None
    Code List: None
    Object Weakref: None
    Guarded Class Weakref: None

  Name: "L['self'].down_proj"
    Source: local_specialized_nn_module
    Create Function: NN_MODULE
    Guard Types: ['ID_MATCH']
    Code List: ["___check_obj_id(L['self'].down_proj, 134798822938000)"]
    Object Weakref: <weakref at 0x7a9582a48f40; to 'Linear' at 0x7a994bf51590>
    Guarded Class Weakref: <weakref at 0x7a983bfd1030; to 'type' at 0x23643c00 (Linear)>

  Name: "G['F']"
    Source: global
    Create Function: FUNCTION_MATCH
    Guard Types: ['ID_MATCH']
    Code List: ["___check_obj_id(G['F'], 134794372950080)"]
    Object Weakref: <weakref at 0x7a9582a488b0; to 'module' at 0x7a9842b7a840>
    Guarded Class Weakref: <weakref at 0x7a994c353a60; to 'type' at 0x91e8e0 (module)>

  Name: ''
    Source: global
    Create Function: DETERMINISTIC_ALGORITHMS
    Guard Types: None
    Code List: None
    Object Weakref: None
    Guarded Class Weakref: None

  Name: "L['self']"
    Source: local
    Create Function: NN_MODULE
    Guard Types: ['ID_MATCH']
    Code List: ["___check_obj_id(L['self'], 134798827026624)"]
    Object Weakref: <weakref at 0x7a9582a48590; to 'MiniFfnBlock' at 0x7a994c3378c0>
    Guarded Class Weakref: <weakref at 0x7a994bfd4130; to 'type' at 0x24619890 (MiniFfnBlock)>

  Name: ''
    Source: global
    Create Function: TORCH_FUNCTION_STATE
    Guard Types: None
    Code List: None
    Object Weakref: None
    Guarded Class Weakref: None

  Name: ''
    Source: shape_env
    Create Function: SHAPE_ENV
    Guard Types: ['SHAPE_ENV', 'SHAPE_ENV', 'SHAPE_ENV', 'SHAPE_ENV', 'SHAPE_ENV', 'SHAPE_ENV', 'SHAPE_ENV']
    Code List: ["L['x'].size()[2] == 1024", "L['x'].stride()[0] == 1024*L['x'].size()[1]", "L['x'].stride()[1] == 1024", "L['x'].stride()[2] == 1", "L['x'].storage_offset() == 0", "2 <= L['x'].size()[0]", "2 <= L['x'].size()[1]"]
    Object Weakref: None
    Guarded Class Weakref: None

  Name: "G['F'].silu"
    Source: global
    Create Function: FUNCTION_MATCH
    Guard Types: ['ID_MATCH']
    Code List: ["___check_obj_id(G['F'].silu, 134794259491040)"]
    Object Weakref: <weakref at 0x7a9582a48ae0; to 'function' at 0x7a983bf468e0>
    Guarded Class Weakref: <weakref at 0x7a994c33ba10; to 'type' at 0x924460 (function)>

  Name: "L['self'].up_proj"
    Source: local_specialized_nn_module
    Create Function: NN_MODULE
    Guard Types: ['ID_MATCH']
    Code List: ["___check_obj_id(L['self'].up_proj, 134798820031664)"]
    Object Weakref: <weakref at 0x7a9582a48db0; to 'Linear' at 0x7a994bc8bcb0>
    Guarded Class Weakref: <weakref at 0x7a983bfd1030; to 'type' at 0x23643c00 (Linear)>

==== Understanding Guards ====

Guards are conditions Dynamo records about your inputs.
If a guard fails on a future call, Dynamo recompiles.

Common guards:
  - tensor.shape[0] == 2        (batch size)
  - tensor.dtype == float16      (data type)
  - tensor.device == cuda:0      (device)
  - tensor.requires_grad == False

In vLLM, guards on batch size are DROPPED (using symbolic shapes)
so that one compiled graph works for all batch sizes.

==== Graph Breaks Demo ====

A 'graph break' happens when Dynamo encounters something it can't trace.

Calling model with graph break:
Exported graph structure:
graph():
    %l_x_ : torch.Tensor [num_users=1] = placeholder[target=L_x_]
    %x : [num_users=1] = call_function[target=operator.mul](args = (%l_x_, 2), kwargs = {})
    return (x,)
  [Backend called] Graph #1 with 3 nodes
    placeholder     | l_x_
    call_function   | x
    output          | output
This print causes a graph break!
Exported graph structure:
graph():
    %l_x_ : torch.Tensor [num_users=1] = placeholder[target=L_x_]
    %x : [num_users=1] = call_function[target=operator.add](args = (%l_x_, 1), kwargs = {})
    return (x,)
  [Backend called] Graph #2 with 3 nodes
    placeholder     | l_x_
    call_function   | x
    output          | output

Dynamo created 2 separate graph(s) due to the print().
Without the print(), it would be 1 graph.
"""