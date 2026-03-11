"""
06 — Tree-Based Speculation and Custom Attention Masks
=====================================================

Tree speculation: instead of a single chain of K tokens,
draft generates a TREE with multiple candidates per position.

This gives the verifier more options — if top-1 draft is wrong,
maybe top-2 is right, salvaging the rest of the tree.

SSD connection: SSD's multi-outcome cache is a special kind of tree.

vLLM code: vllm/v1/attention/backends/tree_attn.py
           vllm/v1/spec_decode/eagle.py -> propose_tree()

Run this: python -m llm.spec_decode.06_tree_attention
"""

import torch


def build_tree_attention_mask(tree_choices: list[tuple[int, ...]], device="cpu"):
    """
    Build tree attention bias from speculative_token_tree config.
    Simplified version of vLLM's _prepare_tree_attn_bias().

    tree_choices: paths in the tree from root.
    - (0,) = first child of root
    - (1,) = second child of root
    - (0,0) = first child of first child
    """
    tree_len = len(tree_choices) + 1  # +1 for root
    mask = torch.full((tree_len, tree_len), float('-inf'), device=device)

    for i in range(tree_len):
        mask[i, i] = 0.0      # attend to self
    mask[:, 0] = 0.0          # all attend to root

    for idx, choice in enumerate(tree_choices):
        node_pos = idx + 1
        for depth in range(1, len(choice)):
            ancestor_path = choice[:depth]
            ancestor_idx = tree_choices.index(ancestor_path) + 1
            mask[node_pos, ancestor_idx] = 0.0

    return mask


def visualize_tree(tree_choices, mask):
    labels = ["root"] + [str(c) for c in tree_choices]
    tree_len = len(labels)

    print(f"\n  Tree: {tree_choices}")
    print(f"  Nodes: {tree_len}\n")

    depths = {}
    for c in tree_choices:
        d = len(c)
        depths.setdefault(d, []).append(str(c))

    print("  Layout:")
    print(f"    Depth 0: root")
    for d in sorted(depths):
        print(f"    Depth {d}: {', '.join(depths[d])}")

    print(f"\n  Attention mask (V=attends, .=masked):")
    header = "           " + " ".join(f"{l:>5}" for l in labels)
    print(header)
    print("          " + "-" * (6 * tree_len + 2))
    for i in range(tree_len):
        cells = ["  V  " if mask[i, j] == 0.0 else "  .  " for j in range(tree_len)]
        print(f"  {labels[i]:>7} |" + "".join(cells))


def main():
    print("=" * 70)
    print("  PART 1: Chain Speculation (Default)")
    print("=" * 70)

    chain = [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]
    visualize_tree(chain, build_tree_attention_mask(chain))
    print("""
    Chain: each node has one child. Standard sequential SD.
    Node (0,0,0) sees root -> (0,) -> (0,0) -> itself.
    """)

    print("=" * 70)
    print("  PART 2: Binary Tree Speculation")
    print("=" * 70)

    binary = [(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]
    visualize_tree(binary, build_tree_attention_mask(binary))
    print("""
    Binary tree: 2 children per node.
    - (0,) and (1,) are siblings -- DON'T see each other
    - (0,0) sees root -> (0,) -> itself (ancestor chain)
    - (1,1) sees root -> (1,) -> itself (different chain)

    If target rejects (0,) "the", maybe (1,) "a" is correct.
    More options = higher chance of acceptance. Cost: more tokens to verify.
    """)

    print("=" * 70)
    print("  PART 3: EAGLE-2 Style Dynamic Tree")
    print("=" * 70)

    eagle2 = [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0)]
    visualize_tree(eagle2, build_tree_attention_mask(eagle2))
    print("""
    EAGLE-2: wider at top (3 candidates), narrower deeper.
    Allocates more budget to promising branches.
    Similar to SSD's geometric fan-out but at DRAFT level.
    """)

    print("=" * 70)
    print("  PART 4: SSD's Multi-Outcome Tree")
    print("=" * 70)
    print("""
    SSD's tree branches at VERIFICATION OUTCOMES, not draft positions:

    Standard tree:
              root
             / | \\
           "the" "a" "an"     <- which token at pos 1?
           / \\
        "cat" "dog"           <- which token at pos 2?

    SSD multi-outcome tree:
              current speculation being verified
             /           |              \\
      (accept 0,     (accept 1,     (accept all,
       bonus="the")   bonus="cat")   bonus="dog")
           |               |              |
      [draft K tokens] [draft K]    [draft K]

    Each SSD branch starts from a DIFFERENT prefix (different
    accepted tokens + different bonus). The attention mask ensures
    each branch sees only its own prefix.

    -> Next: 07_kv_cache_management.py
    """)


if __name__ == "__main__":
    main()