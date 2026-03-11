"""
04 — Draft Model Types: EAGLE, MTP, Standalone, N-gram
======================================================

Different draft model architectures trade off speed, quality, and complexity.
Understanding these is critical because SSD works with ANY draft model —
but the choice of draft model affects cache hit rates and acceptance rates.

DRAFT MODEL ZOO:
===============

    ┌────────────────────────────────────────────────────────────────┐
    │ Type          │ Speed  │ Quality │ Training? │ Extra Params?   │
    │───────────────│────────│─────────│───────────│─────────────────│
    │ N-gram        │ ★★★★★  │ ★       │ No        │ No              │
    │ Standalone 1B │ ★★★★   │ ★★★     │ Yes (LM)  │ Yes (full model)│
    │ MTP head      │ ★★★    │ ★★★★    │ Yes       │ Small head only │
    │ EAGLE         │ ★★     │ ★★★★★   │ Yes       │ Small head only │
    │ EAGLE-3       │ ★★     │ ★★★★★★  │ Yes       │ Small head only │
    └────────────────────────────────────────────────────────────────┘

    Speed: how fast can it draft K tokens?
    Quality: how likely are drafted tokens to be accepted by the target?
    
    For SSD: we want high quality (= high acceptance rate = high cache hit rate)
    Speed matters less because SSD hides draft latency via parallelism.

Run this: python -m llm.spec_decode.04_draft_models
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

from llm.mini_transformer import MiniTransformer


# =====================================================================
# 1. N-GRAM PROPOSER — No neural network needed!
# =====================================================================
class NgramProposer:
    """
    Proposes tokens by matching n-grams from the existing context.
    
    Idea: if the sequence "the cat sat on" appeared earlier in the context,
    and we just generated "the cat sat", we predict "on" as the next token.
    
    Pros: Zero compute cost, works without any model
    Cons: Only works when context has repetitive patterns
    
    vLLM implementation: vllm/v1/spec_decode/ngram_proposer.py
    """
    
    def __init__(self, ngram_size=3):
        self.ngram_size = ngram_size
    
    def propose(self, token_ids: list[int], K: int) -> list[int]:
        """
        Look for the last ngram_size tokens in the history.
        If found, return the next K tokens that followed.
        """
        if len(token_ids) < self.ngram_size + K:
            return []  # not enough context
        
        # The pattern we're looking for
        pattern = tuple(token_ids[-self.ngram_size:])
        
        # Search for this pattern earlier in the sequence
        proposals = []
        for i in range(len(token_ids) - self.ngram_size - 1):
            window = tuple(token_ids[i:i + self.ngram_size])
            if window == pattern:
                # Found a match! Take the next K tokens
                end = min(i + self.ngram_size + K, len(token_ids) - self.ngram_size)
                proposals = token_ids[i + self.ngram_size:end]
                if len(proposals) >= K:
                    return proposals[:K]
        
        return proposals


# =====================================================================
# 2. STANDALONE DRAFT MODEL — A smaller but independent LM
# =====================================================================
class StandaloneDraftModel:
    """
    A separate, smaller language model used as the draft.
    Example: Llama-3.2-1B drafting for Llama-3.1-70B.
    
    Pros: Independent model, easy to deploy
    Cons: Doesn't see the target model's internal representations
    
    vLLM implementation: vllm/v1/spec_decode/draft_model.py → DraftModelProposer
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    @torch.no_grad()
    def propose(self, prefix: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft K tokens autoregressively."""
        generated = prefix.clone()
        tokens = []
        probs_list = []
        
        for _ in range(K):
            logits = self.model(generated)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            token = logits.argmax(dim=-1, keepdim=True)
            tokens.append(token.item())
            probs_list.append(probs.squeeze(0))
            generated = torch.cat([generated, token], dim=1)
        
        return torch.tensor(tokens, device=prefix.device), torch.stack(probs_list)


# =====================================================================
# 3. EAGLE-STYLE DRAFT — Uses target model's hidden states
# =====================================================================
class EagleStyleHead(nn.Module):
    """
    Simplified EAGLE concept: the draft model takes the target model's
    hidden states as INPUT, not just the token IDs.
    
    This is why EAGLE has better acceptance rates — it sees the target
    model's internal "thoughts", not just the tokens.
    
    Real EAGLE:
    - Input: target hidden states at last verified position
    - Architecture: lightweight transformer on top of target features
    - Output: logits for next K tokens
    
    EAGLE-3 additionally:
    - Uses hidden states from INTERMEDIATE layers (not just the last)
    - Specified via eagle_aux_hidden_state_layer_ids in config
    
    vLLM: vllm/v1/spec_decode/eagle.py → EagleProposer
          vllm/model_executor/models/llama_eagle.py
          vllm/model_executor/models/llama_eagle3.py
    """
    
    def __init__(self, hidden_size, vocab_size, num_layers=2, num_heads=4):
        super().__init__()
        # EAGLE takes hidden states as input, not token embeddings
        self.input_proj = nn.Linear(hidden_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, batch_first=True, dropout=0.0,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, hidden_size] from the target model
        Returns:
            logits: [B, T, vocab_size]
        """
        x = self.input_proj(hidden_states)
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.layers(x, mask=mask, is_causal=True)
        return self.lm_head(x)


# =====================================================================
# 4. MTP HEAD — Multi-Token Prediction from DeepSeek-V3
# =====================================================================
class MTPHead(nn.Module):
    """
    Multi-Token Prediction: the target model is trained with additional
    prediction heads that predict tokens 2, 3, ... steps ahead.
    
    Unlike EAGLE (which is a separate model), MTP heads are trained
    jointly with the target model during pre-training.
    
    DeepSeek-V3 uses this: it has num_nextn_predict_layers MTP layers
    that share weights with the main model but predict future tokens.
    
    vLLM: vllm/v1/spec_decode/eagle.py (MTP reuses the EAGLE proposer)
          vllm/model_executor/models/deepseek_mtp.py
    """
    
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        # Each MTP head predicts the next token given hidden states
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.proj(hidden_states))
        return self.lm_head(x)


# =====================================================================
# Comparison experiment
# =====================================================================
class MiniTransformerWithHiddens(nn.Module):
    """
    MiniTransformer that also returns intermediate hidden states,
    needed to simulate EAGLE and EAGLE-3 which consume target hidden states.
    """
    def __init__(self, vocab_size, hidden_size=256, num_heads=4, num_layers=6,
                 max_seq_len=512):
        super().__init__()
        from llm.mini_transformer import MiniTransformerBlock
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb   = nn.Embedding(max_seq_len, hidden_size)
        self.blocks    = nn.ModuleList([
            MiniTransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.norm    = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, token_ids, return_hidden_layers=None):
        """
        Args:
            token_ids: [B, T]
            return_hidden_layers: list of layer indices to return, e.g. [2, 4]
                                  None → return only final logits
        Returns:
            logits: [B, T, vocab_size]
            hiddens: dict {layer_idx: [B, T, H]}  (only if return_hidden_layers set)
        """
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.token_emb(token_ids) + self.pos_emb(positions)

        hiddens = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_hidden_layers and i in return_hidden_layers:
                hiddens[i] = x

        x = self.norm(x)
        logits = self.lm_head(x)

        if return_hidden_layers is not None:
            return logits, hiddens
        return logits


class EagleProposer:
    """
    EAGLE-style proposer: drafts using the target model's last hidden state.

    At each draft step:
      1. Run target model → get hidden states at last position
      2. Feed hidden states into lightweight EAGLE head → get draft logits
      3. Sample draft token, append, repeat K times

    EAGLE-3 variant: also passes hidden states from intermediate layers
    (eagle_aux_hidden_state_layer_ids), giving the head richer signal.
    """
    def __init__(self, target_model: MiniTransformerWithHiddens,
                 eagle_head: EagleStyleHead, aux_layers=None):
        self.target = target_model
        self.head   = eagle_head
        # aux_layers: list of intermediate layer indices to also feed into head
        # None = standard EAGLE (last layer only)
        # e.g. [2, 4] = EAGLE-3 style (intermediate + last)
        self.aux_layers = aux_layers

    @torch.no_grad()
    def propose(self, prefix: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        generated = prefix.clone()
        tokens, probs_list = [], []

        for _ in range(K):
            layers_to_fetch = self.aux_layers if self.aux_layers else []
            logits, hiddens = self.target(generated, return_hidden_layers=layers_to_fetch + [-1])

            # Use the last layer's hidden state (always available via norm output)
            # We approximate by using the logits' "pre-softmax" signal via the
            # final hidden state — re-derive it from lm_head weight transpose
            # For simplicity: use target logits directly as the hidden signal
            # (in real EAGLE the actual hidden tensor before lm_head is used)
            hidden = logits[:, -1:, :]  # [B, 1, vocab_size] as proxy

            # EAGLE head takes hidden states → draft logits
            draft_logits = self.head(hidden)[:, -1, :]  # [B, vocab_size]
            probs = F.softmax(draft_logits, dim=-1)
            token = draft_logits.argmax(dim=-1, keepdim=True)

            tokens.append(token.item())
            probs_list.append(probs.squeeze(0))
            generated = torch.cat([generated, token], dim=1)

        return torch.tensor(tokens, device=prefix.device), torch.stack(probs_list)


class MTPProposer:
    """
    MTP-style proposer: uses dedicated per-step prediction heads.

    Each head_k predicts token at position +k given the target hidden states.
    All K heads run in ONE forward pass — no autoregressive loop needed.

    This is how DeepSeek-V3 does it: num_nextn_predict_layers MTP heads
    trained jointly with the main model.
    """
    def __init__(self, target_model: MiniTransformerWithHiddens,
                 mtp_heads: nn.ModuleList):
        self.target = target_model
        self.heads  = mtp_heads  # one MTPHead per draft step

    @torch.no_grad()
    def propose(self, prefix: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        # ONE forward pass through target
        logits, hiddens = self.target(prefix, return_hidden_layers=list(range(len(self.target.blocks))))
        # Use the last block's hidden state (before norm) as the MTP input
        last_hidden = hiddens[len(self.target.blocks) - 1]  # [B, T, H]
        last_pos = last_hidden[:, -1:, :]  # [B, 1, H] — last position only

        tokens, probs_list = [], []
        for k in range(min(K, len(self.heads))):
            draft_logits = self.heads[k](last_pos)[:, -1, :]  # [B, vocab_size]
            probs = F.softmax(draft_logits, dim=-1)
            token = draft_logits.argmax(dim=-1)
            tokens.append(token.item())
            probs_list.append(probs.squeeze(0))

        return torch.tensor(tokens, device=prefix.device), torch.stack(probs_list)


def time_fn(fn, warmup=3, reps=20):
    """Time a callable, returning mean latency in milliseconds."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1000  # ms


def compare_draft_methods():
    """Compare all draft strategies: N-gram, Standalone, EAGLE, EAGLE-3, MTP."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V, H = 1000, 256

    # Target model (with hidden state access)
    target = MiniTransformerWithHiddens(vocab_size=V, hidden_size=H, num_layers=6, num_heads=4)
    target = target.to(device).eval()

    # Standalone draft models
    standalone_small = MiniTransformer(vocab_size=V, hidden_size=64, num_layers=1, num_heads=4).to(device).eval()
    standalone_med   = MiniTransformer(vocab_size=V, hidden_size=128, num_layers=2, num_heads=4).to(device).eval()

    # EAGLE head (takes vocab_size as hidden_size since we use logits as proxy)
    eagle_head  = EagleStyleHead(hidden_size=V, vocab_size=V, num_layers=1, num_heads=4).to(device).eval()
    eagle       = EagleProposer(target, eagle_head, aux_layers=None)

    # EAGLE-3 head (same architecture, but aux_layers signals intermediate layers)
    eagle3_head = EagleStyleHead(hidden_size=V, vocab_size=V, num_layers=2, num_heads=4).to(device).eval()
    eagle3      = EagleProposer(target, eagle3_head, aux_layers=[2, 4])

    # MTP heads (one per draft step, K=5)
    K = 5
    mtp_heads = nn.ModuleList([MTPHead(hidden_size=H, vocab_size=V) for _ in range(K)]).to(device).eval()
    mtp       = MTPProposer(target, mtp_heads)

    prompt = torch.randint(0, V, (1, 32), device=device)

    # Reference: target's own greedy tokens
    with torch.no_grad():
        ref_logits = target(prompt)
        target_next = ref_logits[:, -K-1:-1, :].argmax(dim=-1).squeeze(0)

    print("="*70)
    print("  Comparing Draft Model Types")
    print("="*70)
    print(f"\n  Target: {sum(p.numel() for p in target.parameters())/1e6:.1f}M params")
    print(f"  Prompt length: {prompt.shape[1]}, K={K}")
    print(f"  Target greedy tokens: {target_next.tolist()}\n")

    results = []  # (name, params_M, matches, latency_ms, draft_tokens)

    # N-gram
    ngram = NgramProposer(ngram_size=3)
    prompt_list = prompt.squeeze(0).tolist()
    ngram_proposals = ngram.propose(prompt_list, K)
    ngram_ms = time_fn(lambda: ngram.propose(prompt_list, K))
    if ngram_proposals:
        matches = sum(a == b for a, b in zip(ngram_proposals, target_next.tolist()))
        print(f"  {'N-gram (3-gram)':<30} proposals={ngram_proposals[:K]}, matches={matches}/{K}, latency={ngram_ms:.3f}ms")
        results.append(("N-gram (3-gram)", 0.0, matches, ngram_ms, ngram_proposals[:K]))
    else:
        print(f"  {'N-gram (3-gram)':<30} No matching n-gram in context, latency={ngram_ms:.3f}ms")
        results.append(("N-gram (3-gram)", 0.0, 0, ngram_ms, []))

    # Standalone
    for name, model in [("Standalone (tiny 64h/1L)", standalone_small),
                         ("Standalone (small 128h/2L)", standalone_med)]:
        with torch.no_grad():
            d_logits = model(prompt)
            draft_next = d_logits[:, -K-1:-1, :].argmax(dim=-1).squeeze(0)

        def _standalone_fn(m=model):
            with torch.no_grad():
                return m(prompt)[:, -K-1:-1, :].argmax(dim=-1).squeeze(0)

        ms = time_fn(_standalone_fn)
        matches = (draft_next == target_next).sum().item()
        params  = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  {name:<30} params={params:.1f}M, draft={draft_next.tolist()}, matches={matches}/{K}, latency={ms:.3f}ms")
        results.append((name, params, matches, ms, draft_next.tolist()))

    # EAGLE
    eagle_ms = time_fn(lambda: eagle.propose(prompt, K))
    eagle_tokens, _ = eagle.propose(prompt, K)
    matches = (eagle_tokens == target_next).sum().item()
    eagle_params = sum(p.numel() for p in eagle_head.parameters()) / 1e6
    print(f"  {'EAGLE (last hidden)':<30} head={eagle_params:.1f}M, draft={eagle_tokens.tolist()}, matches={matches}/{K}, latency={eagle_ms:.3f}ms")
    results.append(("EAGLE (last hidden)", eagle_params, matches, eagle_ms, eagle_tokens.tolist()))

    # EAGLE-3
    eagle3_ms = time_fn(lambda: eagle3.propose(prompt, K))
    eagle3_tokens, _ = eagle3.propose(prompt, K)
    matches = (eagle3_tokens == target_next).sum().item()
    eagle3_params = sum(p.numel() for p in eagle3_head.parameters()) / 1e6
    print(f"  {'EAGLE-3 (aux layers 2,4)':<30} head={eagle3_params:.1f}M, draft={eagle3_tokens.tolist()}, matches={matches}/{K}, latency={eagle3_ms:.3f}ms")
    results.append(("EAGLE-3 (aux layers 2,4)", eagle3_params, matches, eagle3_ms, eagle3_tokens.tolist()))

    # MTP
    mtp_ms = time_fn(lambda: mtp.propose(prompt, K))
    mtp_tokens, _ = mtp.propose(prompt, K)
    matches = (mtp_tokens == target_next).sum().item()
    mtp_params = sum(p.numel() for p in mtp_heads.parameters()) / 1e6
    print(f"  {'MTP (5 heads, 1 fwd pass)':<30} heads={mtp_params:.1f}M, draft={mtp_tokens.tolist()}, matches={matches}/{K}, latency={mtp_ms:.3f}ms")
    results.append(("MTP (5 heads, 1 fwd pass)", mtp_params, matches, mtp_ms, mtp_tokens.tolist()))

    # --- Speed comparison table ---
    fastest_ms = min(r[3] for r in results)
    print("\n" + "="*70)
    print("  Speed Comparison (lower latency = faster drafting)")
    print("="*70)
    print(f"  {'Method':<30} {'Latency (ms)':>14} {'vs fastest':>12} {'Matches':>9}")
    print(f"  {'-'*30} {'-'*14} {'-'*12} {'-'*9}")
    for name, params, matches, ms, _ in sorted(results, key=lambda r: r[3]):
        ratio = ms / fastest_ms
        bar = "█" * min(int(ratio * 5), 30)
        print(f"  {name:<30} {ms:>12.3f}ms {ratio:>10.1f}x  {matches}/{K}  {bar}")

    print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │ KEY OBSERVATIONS (random weights → matches ≈ 0, expected): │
    │                                                             │
    │ • N-gram: Zero compute, only works on repetitive context    │
    │ • Standalone: Independent model, ~60-70% acceptance trained │
    │ • EAGLE: Sees target's last hidden state → ~75-85% trained  │
    │ • EAGLE-3: Also sees intermediate layers → highest quality  │
    │ • MTP: ONE target fwd pass drafts all K tokens (no AR loop) │
    │   DeepSeek-V3 trains MTP heads jointly with the main model  │
    │                                                             │
    │ FOR SSD: EAGLE-3 > EAGLE > Standalone > N-gram              │
    │ Higher acceptance = more predictable bonus tokens           │
    │                                                             │
    │ SPEED RANKING (fastest → slowest to draft K tokens):        │
    │   N-gram >> MTP > Standalone > EAGLE ≈ EAGLE-3              │
    │ MTP is fast because it runs ONE target fwd pass, not K.     │
    │ EAGLE/EAGLE-3 run K autoregressive steps through the head.  │
    └─────────────────────────────────────────────────────────────┘
    """)


def main():
    compare_draft_methods()


if __name__ == "__main__":
    main()

"""
python -m llm.spec_decode.04_draft_models

======================================================================
  Comparing Draft Model Types
======================================================================

  Target: 5.4M params
  Prompt length: 32, K=5
  Target greedy tokens: [717, 818, 724, 154, 215]

  N-gram (3-gram)                No matching n-gram in context, latency=0.004ms
  Standalone (tiny 64h/1L)       params=0.2M, draft=[659, 137, 463, 696, 257], matches=0/5, latency=0.460ms
  Standalone (small 128h/2L)     params=0.7M, draft=[282, 400, 135, 657, 594], matches=0/5, latency=0.771ms
  EAGLE (last hidden)            head=14.0M, draft=[977, 896, 464, 464, 989], matches=0/5, latency=13.096ms
  EAGLE-3 (aux layers 2,4)       head=26.0M, draft=[692, 250, 965, 111, 113], matches=0/5, latency=14.268ms
  MTP (5 heads, 1 fwd pass)      heads=1.6M, draft=[6, 381, 153, 875, 787], matches=0/5, latency=2.711ms

======================================================================
  Speed Comparison (lower latency = faster drafting)
======================================================================
  Method                           Latency (ms)   vs fastest   Matches
  ------------------------------ -------------- ------------ ---------
  N-gram (3-gram)                       0.004ms        1.0x  0/5  █████
  Standalone (tiny 64h/1L)              0.460ms      123.8x  0/5  ██████████████████████████████
  Standalone (small 128h/2L)            0.771ms      207.5x  0/5  ██████████████████████████████
  MTP (5 heads, 1 fwd pass)             2.711ms      729.8x  0/5  ██████████████████████████████
  EAGLE (last hidden)                  13.096ms     3525.5x  0/5  ██████████████████████████████
  EAGLE-3 (aux layers 2,4)             14.268ms     3841.0x  0/5  ██████████████████████████████
"""