"""
08 — Async Scheduling: Overlapping Scheduler + GPU Execution
============================================================

vLLM overlaps CPU scheduling with GPU execution so the GPU never waits.
Understanding this is critical for SSD, which adds ANOTHER layer of
async: draft model overlaps with target verification.

KEY CONCEPTS:
============
1. Sync scheduling: CPU prepares batch -> GPU runs -> CPU processes -> repeat
   (GPU idle while CPU works!)
2. Async scheduling: CPU prepares batch N+1 while GPU runs batch N
   (GPU never idle!)
3. SSD async: draft speculates round T+1 while target verifies round T
   (draft latency hidden!)

vLLM code: vllm/v1/engine/core.py (async scheduling flag)
           vllm/v1/core/sched/async_scheduler.py
           vllm/v1/worker/gpu_model_runner.py (update_async_spec_token_ids)
           docs/design/model_runner_v2.md (MRV2 async-first design)

Run this: python -m llm.spec_decode.08_async_scheduling
"""

import time
import torch
import threading
from collections import deque


# =====================================================================
# PART 1: Synchronous vs Asynchronous Execution
# =====================================================================
def simulate_sync_execution(num_steps=10, cpu_time_ms=2.0, gpu_time_ms=10.0):
    """
    Synchronous: CPU and GPU alternate. GPU idle during CPU work.

    Timeline:
    CPU: [prepare]............[prepare]............[prepare]...
    GPU: .........[execute]...........[execute]...........
         ^                   ^
         GPU waits for CPU   CPU waits for GPU
    """
    total = 0.0
    for _ in range(num_steps):
        time.sleep(cpu_time_ms / 1000)   # CPU prepares
        time.sleep(gpu_time_ms / 1000)   # GPU executes
        total += cpu_time_ms + gpu_time_ms
    return total


def simulate_async_execution(num_steps=10, cpu_time_ms=2.0, gpu_time_ms=10.0):
    """
    Asynchronous: CPU prepares step N+1 while GPU executes step N.

    Timeline:
    CPU: [prep 0][prep 1][prep 2][prep 3]...
    GPU:         [exec 0][exec 1][exec 2]...
                  ^
                  CPU and GPU overlap!

    Total time ≈ max(cpu_time, gpu_time) * num_steps
    (instead of (cpu_time + gpu_time) * num_steps)
    """
    # First step: CPU prepares (no overlap)
    time.sleep(cpu_time_ms / 1000)
    total = cpu_time_ms

    for _ in range(num_steps):
        # GPU executes while CPU prepares next
        gpu_ms = gpu_time_ms
        cpu_ms = cpu_time_ms
        overlap_time = max(gpu_ms, cpu_ms)
        time.sleep(overlap_time / 1000)
        total += overlap_time

    return total


# =====================================================================
# PART 2: The Race Condition Problem
# =====================================================================
def demonstrate_race_condition():
    """
    When CPU and GPU share buffers, async creates race conditions.

    Problem:
      CPU writes to buffer[i] = new_data     (for step N+1)
      GPU reads from buffer[i] = ???          (still executing step N)
      → GPU may read partially-written data!

    vLLM V1 solution: async barrier (synchronize at critical sections)
    vLLM MRV2 solution: separate persistent state from transfer buffers
      - CPU writes to self.states (not pinned, CPU-only)
      - Copy to tmp_states = self.states.pin_memory()
      - GPU reads from tmp_states (safe, CPU won't modify)
    """
    print("""
    RACE CONDITION (the problem):

    Step N:   GPU reads buffer → [0.5, 0.3, 0.2]
    Step N+1: CPU writes buffer → [0.8, ...writing...
    Step N:   GPU reads buffer → [0.8, 0.3, 0.2]  ← CORRUPTED!

    vLLM MRV2 FIX:

    CPU state:     [0.5, 0.3, 0.2]  ← CPU writes here (unpinned)
                        |
                   pin_memory() copy
                        |
    Transfer buf:  [0.5, 0.3, 0.2]  ← GPU reads here (pinned, frozen)
                        |
                   to("cuda", non_blocking=True)
                        |
    GPU tensor:    [0.5, 0.3, 0.2]  ← Model uses this

    CPU can safely write to state while GPU reads from transfer buf.
    """)


# =====================================================================
# PART 3: How Async Scheduling Works with Spec Decode
# =====================================================================
def explain_async_spec_decode():
    """
    The interaction between async scheduling and speculative decoding.
    """
    print("""
    STANDARD SD (synchronous):

    Step N:
      1. CPU: scheduler picks requests, prepares batch
      2. GPU: target model verifies draft tokens from step N-1
      3. GPU: rejection sampling → accepted tokens
      4. GPU: draft model proposes K new tokens
      5. CPU: process outputs, send to clients
      → repeat

    ASYNC SD (vLLM current):

    Step N:
      CPU: prepare batch N+1 (while GPU runs step N)
      GPU: verify step N drafts → reject/accept → draft step N tokens
      Challenge: CPU doesn't know step N results yet when preparing N+1!

    Solution (update_async_spec_token_ids):
      - CPU prepares step N+1 with STALE draft token IDs
      - When step N completes, GPU updates the draft token IDs in-place
      - The stale IDs are only used for scheduling, not for model input

    Code: vllm/v1/worker/gpu_input_batch.py
      def update_async_spec_token_ids(self, draft_token_ids):
          # Called right before model execution to patch in real draft IDs

    SSD ADDS ANOTHER LAYER:

    Step N:
      Target GPU: verify step N drafts
      Draft GPU:  SIMULTANEOUSLY build speculation cache for step N+1
      On verify complete:
        Cache hit → instant! Send cached speculation to target
        Cache miss → fallback: draft just-in-time (or use fast n-gram)

    The key: SSD's async is at the HARDWARE level (separate GPUs),
    while vLLM's async scheduling is at the CPU/GPU level.
    They compose naturally — MRV2's async-first design is a perfect fit.
    """)


def main():
    print("=" * 70)
    print("  PART 1: Sync vs Async Execution")
    print("=" * 70)

    cpu_ms, gpu_ms = 2.0, 10.0
    num_steps = 10

    sync_time = simulate_sync_execution(num_steps, cpu_ms, gpu_ms)
    async_time = simulate_async_execution(num_steps, cpu_ms, gpu_ms)

    print(f"\n  CPU time per step: {cpu_ms}ms, GPU time per step: {gpu_ms}ms")
    print(f"  Sync total:  {sync_time:.0f}ms  ({num_steps} * ({cpu_ms}+{gpu_ms}))")
    print(f"  Async total: {async_time:.0f}ms  ({num_steps} * max({cpu_ms},{gpu_ms}) + {cpu_ms})")
    print(f"  Speedup: {sync_time/async_time:.2f}x")

    print("""
    With cpu_ms << gpu_ms, async nearly eliminates CPU overhead.
    GPU utilization goes from ~83% to ~100%.
    """)

    print("=" * 70)
    print("  PART 2: The Race Condition Problem")
    print("=" * 70)
    demonstrate_race_condition()

    print("=" * 70)
    print("  PART 3: Async Scheduling + Spec Decode + SSD")
    print("=" * 70)
    explain_async_spec_decode()

    print("""
    KEY TAKEAWAY for SSD implementation:

    MRV2's async-first design means:
    1. No CPU sync points in the main loop → NCCL can be async too
    2. GPU-native input prep (Triton) → cache hit/miss decision on GPU
    3. StagedWriteTensor → speculation cache updates without full copies
    4. Explicit CUDA graphs → draft model forward pass can be captured

    SSD fits naturally into MRV2 because both are async-first.
    The draft worker runs its own CUDA stream on its own GPU,
    communicates via async NCCL, and the main loop never blocks.

    -> Next: 09_nccl_communication.py
    """)


if __name__ == "__main__":
    main()