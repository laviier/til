"""
05 — Parallel Drafting: Generate All K Tokens in One Pass
=========================================================

Sequential drafting: K forward passes → K tokens (slow)
Parallel drafting:   1 forward pass  → K tokens (fast!)

This is the KEY building block for SSD. SSD extends parallel drafting
to draft for MULTIPLE verification outcomes simultaneously.

HOW IT WORKS:
============

    Sequential (standard EAGLE):
    ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
    │ Draft  │→ │ Draft  │→ │ Draft  │→ │ Draft  │→ │ Draft  │
    │ tok 1  │  │ tok 2  │  │ tok 3  │  │ tok 4  │  │ tok 5  │
    └────────┘  └────────┘  └────────┘  └────────┘  └────────┘
    5 forward passes, each sees previous draft tokens

    Parallel:
    ┌──────────────────────────────────────────────────┐
    │ Draft tok 1, tok 2, tok 3, tok 4, tok 5          │
    │ (all in ONE forward pass with masked attention)   │
    └──────────────────────────────────────────────────┘
    1 forward pass, each position sees ONLY the verified prefix

    The trick: use a special attention mask where draft positions
    can attend to the prefix but NOT to each other.

vLLM config: speculative_config.parallel_drafting = True
vLLM code:   SpecDecodeBaseProposer.set_inputs_first_pass() (else branch)

Run this: python -m llm.spec_decode.05_parallel_drafting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from llm.mini_transformer import MiniTransformer


def build_parallel_draft_mask(prefix_len: int, K: int) -> torch.Tensor:
    """
    Build the attention mask for parallel drafting.

    Rules:
    - Prefix positions: standard causal mask
    - Draft positions: each sees ALL prefix tokens + self, but NOT other drafts

    Example with prefix_len=3, K=4:

        Positions:  p0  p1  p2 | d0  d1  d2  d3
                   ─────────────┼────────────────
        p0          ✓   ·   ·    ·   ·   ·   ·
        p1          ✓   ✓   ·    ·   ·   ·   ·
        p2          ✓   ✓   ✓    ·   ·   ·   ·
        ────────────────────────┼────────────────
        d0          ✓   ✓   ✓    ✓   ·   ·   ·
        d1          ✓   ✓   ✓    ·   ✓   ·   ·
        d2          ✓   ✓   ✓    ·   ·   ✓   ·
        d3          ✓   ✓   ✓    ·   ·   ·   ✓
    """
    total = prefix_len + K
    mask = torch.full((total, total), float('-inf'))

    # Prefix: standard causal (lower triangular)
    for i in range(prefix_len):
        mask[i, :i+1] = 0.0

    # Draft positions: see all prefix tokens + self
    for i in range(K):
        row = prefix_len + i
        mask[row, :prefix_len] = 0.0
        mask[row, row] = 0.0

    return mask


def visualize_mask(mask: torch.Tensor, prefix_len: int, K: int):
    """Print the attention mask in a readable format."""
    total = prefix_len + K
    labels = [f"p{i}" for i in range(prefix_len)] + [f"d{i}" for i in range(K)]
    header = "         " + " ".join(f"{l:>3}" for l in labels)
    print(header)
    print("        " + "-" * (4 * total + 4))

    for i in range(total):
        cells = []
        for j in range(total):
            cells.append(" V " if mask[i, j] == 0.0 else " . ")
        sep = " |" if i == prefix_len else "  "
        print(f"  {labels[i]:>4}{sep}" + "".join(cells))
        if i == prefix_len - 1:
            print("        " + "-" * (4 * total + 4))


@torch.no_grad()
def parallel_draft(model, prefix_ids: torch.Tensor, K: int, pard_token_id: int = 0):
    """Generate K draft tokens in ONE forward pass using masked attention."""
    device = prefix_ids.device
    B, prefix_len = prefix_ids.shape

    pard_tokens = torch.full((B, K), pard_token_id, device=device, dtype=prefix_ids.dtype)
    input_ids = torch.cat([prefix_ids, pard_tokens], dim=1)

    mask = build_parallel_draft_mask(prefix_len, K).to(device)
    total = prefix_len + K
    positions = torch.arange(total, device=device).unsqueeze(0)

    # Run model with custom mask (access internals of MiniTransformer)
    x = model.token_emb(input_ids) + model.pos_emb(positions)
    for block in model.blocks:
        residual = x
        x_norm = block.attn.norm(x)
        B_s, T, H = x_norm.shape
        nh, hd = block.attn.num_heads, block.attn.head_dim

        q = block.attn.q_proj(x_norm).view(B_s, T, nh, hd).transpose(1, 2)
        k = block.attn.k_proj(x_norm).view(B_s, T, nh, hd).transpose(1, 2)
        v = block.attn.v_proj(x_norm).view(B_s, T, nh, hd).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B_s, T, H)
        x = block.attn.out_proj(out) + residual
        x = block.ffn(x)

    x = model.norm(x)
    logits = model.lm_head(x)
    draft_logits = logits[0, prefix_len:prefix_len+K, :]
    return draft_logits.argmax(dim=-1)


@torch.no_grad()
def sequential_draft(model, prefix_ids: torch.Tensor, K: int):
    """Standard sequential drafting: K forward passes."""
    generated = prefix_ids.clone()
    tokens = []
    for _ in range(K):
        logits = model(generated)[:, -1, :]
        tok = logits.argmax(dim=-1, keepdim=True)
        tokens.append(tok.item())
        generated = torch.cat([generated, tok], dim=1)
    return torch.tensor(tokens, device=prefix_ids.device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V, H, K = 100, 128, 5

    print("=" * 70)
    print("  PART 1: The Parallel Drafting Attention Mask")
    print("=" * 70)

    mask = build_parallel_draft_mask(prefix_len=4, K=3)
    print("\n  Attention mask for prefix_len=4, K=3:")
    print("  (V = can attend, . = masked)\n")
    visualize_mask(mask, prefix_len=4, K=3)

    print("""
    Key observation: draft positions d0, d1, d2 each see the ENTIRE prefix
    but NOT each other. Each independently predicts a future token given
    only the verified prefix context.

    This means parallel drafting tokens are LESS informed than sequential:
    - Sequential d2 sees prefix + d0 + d1
    - Parallel d2 sees only prefix

    -> Parallel has LOWER acceptance rate but is MUCH faster (1 pass vs K).
    """)

    print("=" * 70)
    print("  PART 2: Sequential vs Parallel Drafting")
    print("=" * 70)

    model = MiniTransformer(vocab_size=V, hidden_size=H, num_layers=2, num_heads=4)
    model = model.to(device).eval()
    prompt = torch.randint(0, V, (1, 16), device=device)

    seq_tokens = sequential_draft(model, prompt, K)
    par_tokens = parallel_draft(model, prompt, K)

    print(f"\n  Sequential draft: {seq_tokens.tolist()}")
    print(f"  Parallel draft:   {par_tokens.tolist()}")
    matches = (seq_tokens == par_tokens).sum().item()
    print(f"  Agreement: {matches}/{K} tokens match")

    print("""
    The first token should match (both see the same prefix).
    Later tokens may differ because sequential sees previous drafts.

    In practice with trained models:
    - Sequential: ~75-85% acceptance rate
    - Parallel:   ~60-70% (lower, but 1 forward pass!)
    """)

    print("=" * 70)
    print("  PART 3: Connection to SSD")
    print("=" * 70)
    print("""
    SSD extends parallel drafting to handle MULTIPLE outcomes:

    Standard parallel drafting:
      [prefix] -> [draft_pos_1, ..., draft_pos_K]
      One set of K positions, all seeing only the prefix.

    SSD multi-outcome drafting:
      For each possible verification outcome (k_accepted, bonus_token):
        [prefix + accepted_tokens + bonus] -> [draft_pos_1, ..., draft_pos_K]

      Multiple sets of K positions, each seeing a DIFFERENT prefix.

    The SSD attention mask (Figure 8 in paper) is an extension of the
    parallel drafting mask -- each "branch" is an independent parallel draft.

    Total tokens decoded in one forward pass: B x F x (K+1)
    where B = batch size, F = fan-out, K = speculation length.

    -> Next: 06_tree_attention.py
    """)


if __name__ == "__main__":
    main()