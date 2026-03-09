"""
08 — Async Scheduling: Overlapping Scheduler + GPU Execution
=============================================================

Understanding vLLM's async scheduling — critical for SSD's async overlap.
EXERCISES:
1. Understand the async scheduling barrier
2. Trace how draft token IDs are updated asynchronously
3. Understand MRV2's async-first design (no CPU sync points)
4. KEY FOR SSD: SSD adds another level of async — draft overlaps with verify
5. Implement a toy async producer-consumer with CUDA streams

Next: 09_*.py
"""

# TODO: Implement exercises above
# Follow the pattern from 00-02 for structure

def main():
    print(__doc__)
    print("TODO: Implement the exercises in this file!")
    print("See the docstring above for detailed instructions.")

if __name__ == "__main__":
    main()
