"""
09 — NCCL Communication Benchmarks for Disaggregation
======================================================

Measure the communication overhead that SSD introduces.
EXERCISES:
1. Setup torch.distributed with 2+ GPUs
2. Benchmark point-to-point send/recv latency (small payloads: verification outcome)
3. Benchmark medium payloads (draft tokens + logits)
4. Compare NVLink vs PCIe bandwidth
5. Compute: communication overhead / verification latency ratio
6. KEY: if overhead < 5%, NCCL is NOT the bottleneck for SSD

Next: 10_*.py
"""

# TODO: Implement exercises above
# Follow the pattern from 00-02 for structure

def main():
    print(__doc__)
    print("TODO: Implement the exercises in this file!")
    print("See the docstring above for detailed instructions.")

if __name__ == "__main__":
    main()
