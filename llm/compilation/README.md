# LLM Compilation — Hands-On Learning

Breaking down the full GPU compilation stack, one script at a time.
Each script covers one layer. Write them yourself from scratch.

## Layers to cover

0. `00_eager_baseline.py` — Eager PyTorch, measuring per-op overhead
1. `01_dynamo_graph_capture.py` — TorchDynamo: graph export, guards, graph breaks
2. `02_fx_graph.py` — FX Graph: nodes, transformations, fusion passes
3. `03_inductor.py` — Inductor: compiling, inspecting generated code, benchmarking
4. `04_triton_kernels.py` — Triton: write your first GPU kernels
5. `05_ptx_inspection.py` — PTX/SASS: extracting and reading GPU assembly
6. `06_gpu_hardware.py` — Measuring bandwidth, TFLOPS, launch overhead
7. `07_kernel_zoo.py` — cuBLAS vs CUTLASS vs Triton benchmarks
8. `08_cuda_graphs.py` — CUDA Graph capture and replay
9. `09_full_pipeline.py` — Everything together, end-to-end benchmark
