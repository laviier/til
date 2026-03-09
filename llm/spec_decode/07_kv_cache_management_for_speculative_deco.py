"""
07 — KV Cache Management for Speculative Decoding
==================================================

The hardest systems challenge: managing KV cache with rollbacks.
EXERCISES:
1. Understand PagedAttention and block tables
2. Implement KV cache rollback on rejection (discard rejected token KV)
3. Trace vLLM's block table reconciliation after verification
4. KEY FOR SSD: draft model has its OWN KV cache on a separate GPU
5. Implement dual-device block table management

Next: 08_*.py
"""

# TODO: Implement exercises above
# Follow the pattern from 00-02 for structure

def main():
    print(__doc__)
    print("TODO: Implement the exercises in this file!")
    print("See the docstring above for detailed instructions.")

if __name__ == "__main__":
    main()
