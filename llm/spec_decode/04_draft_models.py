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
    │ N-gram        │ ★★★★★ │ ★       │ No        │ No              │
    │ Standalone 1B │ ★★★★  │ ★★★     │ Yes (LM)  │ Yes (full model)│
    │ MTP head      │ ★★★   │ ★★★★    │ Yes       │ Small head only │
    │ EAGLE         │ ★★    │ ★★★★★   │ Yes       │ Small head only │
    │ EAGLE-3       │ ★★    │ ★★★★★★  │ Yes       │ Small head only │
    └────────────────────────────────────────────────────────────────┘

    Speed: how fast can it draft K tokens?
    Quality: how likely are drafted tokens to be accepted by the target?
    
    For SSD: we want high quality (= high acceptance rate = high cache hit rate)
    Speed matters less because SSD hides draft latency via parallelism.

Run this: python -m llm.spec_decode.04_draft_models
"""

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
def compare_draft_methods():
    """Compare acceptance rates of different draft strategies."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = 1000
    H = 256
    
    # Create a "target" model
    target = MiniTransformer(vocab_size=V, hidden_size=H, num_layers=6, num_heads=4)
    target = target.to(device).eval()
    
    # Create draft models of different types
    standalone_small = MiniTransformer(vocab_size=V, hidden_size=64, num_layers=1, num_heads=4)
    standalone_small = standalone_small.to(device).eval()
    
    standalone_med = MiniTransformer(vocab_size=V, hidden_size=128, num_layers=2, num_heads=4)
    standalone_med = standalone_med.to(device).eval()
    
    print("="*70)
    print("  Comparing Draft Model Types")
    print("="*70)
    
    # Generate reference output from target
    prompt = torch.randint(0, V, (1, 32), device=device)
    K = 5
    
    # Get target's greedy tokens for reference
    with torch.no_grad():
        target_logits = target(prompt)
        target_next = target_logits[:, -K-1:-1, :].argmax(dim=-1).squeeze(0)  # last K positions
    
    methods = {
        "N-gram (3-gram)": None,  # handled separately
        "Standalone (tiny, 64h/1L)": standalone_small,
        "Standalone (small, 128h/2L)": standalone_med,
    }
    
    print(f"\n  Target model: {sum(p.numel() for p in target.parameters())/1e6:.1f}M params")
    print(f"  Prompt length: {prompt.shape[1]}, K={K} draft tokens")
    print(f"  Target's greedy tokens: {target_next.tolist()}\n")
    
    # N-gram test
    prompt_list = prompt.squeeze(0).tolist()
    ngram = NgramProposer(ngram_size=3)
    ngram_proposals = ngram.propose(prompt_list, K)
    if ngram_proposals:
        matches = sum(1 for a, b in zip(ngram_proposals, target_next.tolist()) if a == b)
        print(f"  N-gram (3-gram):     proposals={ngram_proposals[:K]}, "
              f"greedy matches={matches}/{min(K, len(ngram_proposals))}")
    else:
        print(f"  N-gram (3-gram):     No matching n-gram found in context")
    
    # Standalone models
    for name, model in methods.items():
        if model is None:
            continue
        with torch.no_grad():
            draft_logits = model(prompt)
            draft_next = draft_logits[:, -K-1:-1, :].argmax(dim=-1).squeeze(0)
        
        matches = (draft_next == target_next).sum().item()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  {name}: params={params:.1f}M, "
              f"draft={draft_next.tolist()}, greedy matches={matches}/{K}")
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │ KEY OBSERVATIONS:                                           │
    │                                                             │
    │ With RANDOM weights (untrained), acceptance rates are ~0.   │
    │ This is expected! In practice:                              │
    │                                                             │
    │ • N-gram: Great for repetitive text (code, templates)       │
    │   but useless for novel content. Zero compute cost.         │
    │                                                             │
    │ • Standalone: Needs training on same data as target.        │
    │   Llama-3.2-1B for Llama-3.1-70B gets ~60-70% acceptance.  │
    │                                                             │
    │ • EAGLE: Higher acceptance (~75-85%) because it sees        │
    │   target hidden states. But needs separate training.        │
    │                                                             │
    │ • EAGLE-3: Even higher acceptance by using intermediate     │
    │   layer hidden states from the target model.                │
    │                                                             │
    │ • MTP: Trained jointly with target, no separate training    │
    │   needed. DeepSeek-V3 natively includes MTP heads.          │
    │                                                             │
    │ FOR SSD: Higher acceptance rate → higher cache hit rate     │
    │ → more effective speculation cache. EAGLE-3 is the best     │
    │ draft model for SSD (highest acceptance = most predictable  │
    │ bonus tokens).                                              │
    └─────────────────────────────────────────────────────────────┘
    """)


def main():
    compare_draft_methods()


if __name__ == "__main__":
    main()