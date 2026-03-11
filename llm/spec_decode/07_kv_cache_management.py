"""
07 — KV Cache Management for Speculative Decoding
==================================================

The hardest systems challenge: managing KV cache when tokens get rejected.

Rejected draft tokens wrote KV entries that are now WRONG and must be
rolled back. With paged memory (PagedAttention), this means tracking
which blocks/slots to invalidate.

For SSD: the draft model has its OWN KV cache on a SEPARATE GPU.

vLLM code: vllm/v1/core/kv_cache_manager.py
           vllm/v1/core/single_type_kv_cache_manager.py

Run this: python -m llm.spec_decode.07_kv_cache_management
"""

import torch


class SimplePagedKVCache:
    """Simplified paged KV cache to understand the concept."""

    def __init__(self, num_blocks=16, block_size=4, num_heads=4, head_dim=32):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.key_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        self.value_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        self.free_blocks = list(range(num_blocks))
        self.allocated = {}   # request_id -> list of block_ids
        self.seq_lens = {}    # request_id -> current seq len

    def allocate(self, req_id: str, num_tokens: int) -> list[int]:
        n = (num_tokens + self.block_size - 1) // self.block_size
        blocks = [self.free_blocks.pop(0) for _ in range(n)]
        self.allocated[req_id] = blocks
        self.seq_lens[req_id] = 0
        return blocks

    def write_kv(self, req_id: str, pos: int, k: torch.Tensor, v: torch.Tensor):
        blocks = self.allocated[req_id]
        bid = blocks[pos // self.block_size]
        slot = pos % self.block_size
        self.key_cache[bid, slot] = k
        self.value_cache[bid, slot] = v
        self.seq_lens[req_id] = max(self.seq_lens[req_id], pos + 1)

    def slot_mapping(self, req_id: str, positions: list[int]) -> list[int]:
        blocks = self.allocated[req_id]
        return [blocks[p // self.block_size] * self.block_size + p % self.block_size
                for p in positions]

    def rollback(self, req_id: str, new_len: int):
        """Rollback: discard KV entries for rejected tokens."""
        old_len = self.seq_lens[req_id]
        if new_len >= old_len:
            return

        blocks = self.allocated[req_id]
        for pos in range(new_len, old_len):
            bid = blocks[pos // self.block_size]
            slot = pos % self.block_size
            self.key_cache[bid, slot] = 0
            self.value_cache[bid, slot] = 0

        new_nblocks = (new_len + self.block_size - 1) // self.block_size
        if new_nblocks < len(blocks):
            freed = blocks[new_nblocks:]
            self.free_blocks.extend(freed)
            self.allocated[req_id] = blocks[:new_nblocks]

        self.seq_lens[req_id] = new_len

    def status(self, req_id: str) -> str:
        blocks = self.allocated.get(req_id, [])
        sl = self.seq_lens.get(req_id, 0)
        return (f"  '{req_id}': seq_len={sl}, blocks={blocks}, "
                f"free={len(self.free_blocks)}/{self.num_blocks}")


def main():
    print("=" * 70)
    print("  PART 1: Basic Paged KV Cache")
    print("=" * 70)

    cache = SimplePagedKVCache(num_blocks=8, block_size=4, num_heads=2, head_dim=16)
    blocks = cache.allocate("req1", num_tokens=10)
    print(f"\n  Allocated blocks for 10 tokens: {blocks}")
    print(f"  Block size=4, so need ceil(10/4)=3 blocks")

    for pos in range(8):
        cache.write_kv("req1", pos, torch.randn(2, 16), torch.randn(2, 16))

    print(f"\n  After writing 8 tokens:")
    print(cache.status("req1"))
    slots = cache.slot_mapping("req1", list(range(8)))
    print(f"  Slot mapping [0-7] -> {slots}")

    print("\n" + "=" * 70)
    print("  PART 2: Rollback on Rejection")
    print("=" * 70)
    print("""
    Scenario: drafted 5 tokens (pos 3-7), target rejects at pos 5.

    Before:  pos: 0  1  2  3  4  5  6  7
             src: [prefix ] [draft          ]

    After:   pos: 0  1  2  3  4  5
             src: [prefix ] [acc] [bonus]

    Positions 6, 7 must be ROLLED BACK from KV cache.
    """)

    print(f"  Before rollback:")
    print(cache.status("req1"))
    cache.rollback("req1", new_len=6)
    print(f"\n  After rollback to seq_len=6:")
    print(cache.status("req1"))

    print("\n" + "=" * 70)
    print("  PART 3: Why This Matters for SSD")
    print("=" * 70)
    print("""
    Standard SD (co-located):
      - Draft and target share the SAME KV cache
      - On rejection: rollback both draft and target KV entries
      - Simple: same device, same memory

    SSD (disaggregated):
      - Draft has its OWN KV cache on a SEPARATE GPU
      - Target has its own KV cache on the TP group
      - On cache hit: draft KV is already valid (pre-computed)
      - On cache miss: draft must rebuild KV from scratch
      - NO KV transfer between devices (too expensive)

    The SSD paper (Section B.1) describes this:
      "The target model never sees the draft-side speculation cache.
       No KV cache is ever transferred between the devices."

    Key challenge for implementation:
      1. Target side: normal rollback (same as standard SD)
      2. Draft side: must maintain KV cache for ALL speculation branches
         in the cache, then discard non-matching branches after verify
      3. After verification, draft extends the winning branch's KV
         via an "extend" (prefill) operation before async decoding

    vLLM code to study:
      - vllm/v1/core/kv_cache_manager.py (block allocation)
      - vllm/v1/core/single_type_kv_cache_manager.py (rollback logic)
      - PADDING_SLOT_ID in vllm/v1/spec_decode/utils.py (invalid slots)

    -> Next: 08_async_scheduling.py
    """)


if __name__ == "__main__":
    main()