# Speculative Decoding — Hands-On Learning

Building up deep understanding of vLLM speculative decoding and parallel drafting,
one script at a time. Write them yourself from scratch.

Prerequisite: Complete the `compilation/` series first.

## Layers to cover

0. `00_ar_baseline.py` — Autoregressive decoding: the bottleneck we're solving
1. `01_spec_decode_from_scratch.py` — Implement speculative decoding from first principles
2. `02_rejection_sampling.py` — The math: acceptance rates, residual distributions, bonus tokens
3. `03_vllm_sd_tracing.py` — Trace vLLM's speculative decoding step-by-step
4. `04_draft_models.py` — Draft model types: EAGLE, MTP, standalone, n-gram
5. `05_parallel_drafting.py` — Parallel drafting: generate all K tokens in one pass
6. `06_tree_attention.py` — Tree-based speculation and custom attention masks
7. `07_kv_cache_management.py` — KV cache allocation, rollback, and reconciliation for SD
8. `08_async_scheduling.py` — How async scheduling overlaps scheduler + GPU execution
9. `09_nccl_communication.py` — Device-to-device communication benchmarks for disaggregation
10. `10_speculation_cache.py` — SSD core: predicting verification outcomes and caching speculations

## Learning Path

```
00-02: Foundations (understand SD from scratch)
   │
   ▼
03-04: vLLM internals (trace the real code)
   │
   ▼
05-06: Advanced drafting (parallel + tree — needed for SSD)
   │
   ▼
07-08: Systems (KV cache + async — the hard parts of integration)
   │
   ▼
09-10: SSD-specific (disaggregation + speculation cache)
```
