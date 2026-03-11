"""
09 — NCCL Communication Benchmarks for Disaggregation
=====================================================

SSD puts the draft model on a SEPARATE GPU. Target and draft communicate
via NCCL (NVIDIA Collective Communication Library).

This script benchmarks the communication overhead to determine if
NCCL is a bottleneck for SSD. This is POC Day 4 validation.

WHAT SSD COMMUNICATES:
=====================
  Target -> Draft: verification outcome
    Payload: (k_accepted: int32, bonus_token: int32) * batch_size
    Size: ~8 bytes * BS (tiny!)

  Draft -> Target: speculated tokens + logits for verification
    Payload: K * (token_id: int32 + logits: float32[V]) * batch_size
    Size: K * (4 + V*4) * BS
    For K=5, V=128000, BS=1: ~2.5 MB

GO/NO-GO: if round-trip < 5% of verification time, NCCL is not the bottleneck.

REQUIREMENTS: 2+ GPUs with NCCL support
If you don't have multiple GPUs, this script shows the expected numbers.

Run this: torchrun --nproc_per_node=2 -m llm.spec_decode.09_nccl_communication
Or for single-GPU: python -m llm.spec_decode.09_nccl_communication
"""

import os
import time
import torch


def benchmark_nccl_p2p(rank, world_size, device):
    """Benchmark point-to-point NCCL send/recv."""
    results = {}

    for name, size_bytes in [
        ("Verify outcome (8B)", 8),
        ("Verify outcome BS=16 (128B)", 128),
        ("Draft tokens K=5 (20B)", 20),
        ("Draft tokens+logits K=5 V=1000 (20KB)", 20_000),
        ("Draft tokens+logits K=5 V=32000 (640KB)", 640_000),
        ("Draft tokens+logits K=5 V=128000 (2.5MB)", 2_500_000),
    ]:
        num_elements = max(1, size_bytes // 4)
        tensor = torch.randn(num_elements, device=device)

        # Warmup
        for _ in range(5):
            if rank == 0:
                torch.distributed.send(tensor, dst=1)
                torch.distributed.recv(tensor, src=1)
            else:
                torch.distributed.recv(tensor, src=0)
                torch.distributed.send(tensor, dst=0)

        # Benchmark
        torch.cuda.synchronize()
        num_iters = 100
        start = time.perf_counter()
        for _ in range(num_iters):
            if rank == 0:
                torch.distributed.send(tensor, dst=1)
                torch.distributed.recv(tensor, src=1)
            else:
                torch.distributed.recv(tensor, src=0)
                torch.distributed.send(tensor, dst=0)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iters

        results[name] = elapsed * 1000  # ms
        if rank == 0:
            print(f"    {name:45s}: {elapsed*1000:.3f} ms round-trip")

    return results


def run_distributed():
    """Run with torchrun for actual NCCL benchmarks."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.distributed.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print("=" * 70)
        print("  NCCL Point-to-Point Benchmarks")
        print("=" * 70)
        print(f"\n  World size: {world_size}, Backend: NCCL\n")

    results = benchmark_nccl_p2p(rank, world_size, device)

    if rank == 0:
        print(f"""
    ANALYSIS for SSD:

    Typical verification time for 70B model (TP=4): ~15-25 ms
    NCCL round-trip for verify outcome (8B):   {results.get('Verify outcome (8B)', 0.05):.3f} ms
    NCCL round-trip for draft+logits (2.5MB):  {results.get('Draft tokens+logits K=5 V=128000 (2.5MB)', 0.5):.3f} ms

    Total SSD communication per round: ~{results.get('Verify outcome (8B)', 0.05) + results.get('Draft tokens+logits K=5 V=128000 (2.5MB)', 0.5):.3f} ms
    As fraction of verify time (~20ms):  ~{(results.get('Verify outcome (8B)', 0.05) + results.get('Draft tokens+logits K=5 V=128000 (2.5MB)', 0.5))/20*100:.1f}%

    OPTIMIZATION: Instead of sending full V-dim logits, send only top-K
    logits needed for rejection sampling. This reduces the draft->target
    payload from 2.5MB to ~20KB.
        """)

    torch.distributed.destroy_process_group()


def show_expected_numbers():
    """Show expected numbers when multi-GPU is not available."""
    print("=" * 70)
    print("  NCCL Communication — Expected Numbers (no multi-GPU available)")
    print("=" * 70)
    print("""
    On a typical 8xH100 node with NVLink:

    ┌─────────────────────────────────────────────────────┐
    │ Payload              │ Size    │ Round-trip Latency  │
    │──────────────────────│─────────│─────────────────────│
    │ Verify outcome       │ 8 B     │ ~0.01-0.05 ms      │
    │ Verify outcome BS=16 │ 128 B   │ ~0.01-0.05 ms      │
    │ Draft tokens K=5     │ 20 B    │ ~0.01-0.05 ms      │
    │ Draft+logits (V=32K) │ 640 KB  │ ~0.1-0.3 ms        │
    │ Draft+logits (V=128K)│ 2.5 MB  │ ~0.3-0.8 ms        │
    └─────────────────────────────────────────────────────┘

    NVLink bandwidth: 900 GB/s (H100)
    PCIe Gen5 bandwidth: 128 GB/s

    For 2.5 MB over NVLink: 2.5MB / 900 GB/s = 0.003 ms (bandwidth)
    But latency dominates for small payloads: ~0.01-0.05 ms minimum

    COMPARISON with verification time:
    - 70B model verification: ~15-25 ms
    - Total NCCL overhead: ~0.3-0.8 ms (with full logits)
    - As % of verify time: 1.5-4% ← NEGLIGIBLE

    OPTIMIZATION: send only top-K logits instead of full vocab:
    - Top-256 logits: 256 * 4 * 5 = 5 KB → ~0.05 ms
    - Overhead drops to < 0.5%

    CONCLUSION: NCCL is NOT the bottleneck for SSD.
    The paper confirms this (Section C): "communications are not a
    bottleneck in practice."
    """)

    print("=" * 70)
    print("  SSD Communication Protocol")
    print("=" * 70)
    print("""
    Per speculation round:

    1. Target -> Draft (after verification):
       Send: {seq_id, k_accepted, bonus_token, temperature} per request
       Size: ~16 bytes * batch_size
       Timing: immediately after rejection sampling

    2. Draft -> Target (from speculation cache or fallback):
       Send: {K draft_token_ids, K * top_N logits} per request
       Size: ~K * (4 + N*4) * batch_size (N = top-N for verification)
       Timing: immediately (cache hit) or after fallback drafting

    The key insight: SSD needs only ONE round-trip per speculation round,
    and both payloads are small enough that NVLink latency dominates.

    -> Next: 10_speculation_cache.py (the core of SSD!)
    """)


def main():
    if torch.distributed.is_available() and "RANK" in os.environ:
        run_distributed()
    else:
        print("  (No distributed environment detected. Showing expected numbers.)")
        print("  (To run actual benchmarks: torchrun --nproc_per_node=2 -m llm.spec_decode.09_nccl_communication)\n")
        show_expected_numbers()


if __name__ == "__main__":
    main()