"""
01 — Speculative Decoding from First Principles
================================================

Implement the full speculative decoding algorithm from scratch.
NO vLLM dependencies — pure PyTorch.

THE ALGORITHM:
=============

    ┌─────────────────────────────────────────────────────────────────┐
    │ Speculative Decoding Loop                                       │
    │                                                                  │
    │  1. DRAFT: Small model generates K tokens autoregressively      │
    │     draft = [d1, d2, d3, d4, d5]  (fast, K forward passes)     │
    │                                                                  │
    │  2. VERIFY: Large model scores ALL K tokens in ONE forward pass │
    │     target_logits = target_model([prefix, d1, d2, d3, d4, d5]) │
    │     (This costs almost the same as generating 1 token!)         │
    │                                                                  │
    │  3. REJECT: For each position, accept with probability          │
    │     min(1, p_target(d_i) / p_draft(d_i))                       │
    │     On first rejection at position k:                           │
    │       - Discard d_{k+1}, ..., d_K                               │
    │       - Sample bonus token from residual distribution           │
    │                                                                  │
    │  4. RESULT: Accept tokens [d1, ..., d_k, bonus]                 │
    │     Expected accepted: α*K + 1 tokens per round                 │
    │     (α = acceptance rate, depends on draft quality)              │
    └─────────────────────────────────────────────────────────────────┘

EXERCISES:
=========
1. Implement draft_tokens() — K sequential forward passes of draft model
2. Implement verify_tokens() — ONE forward pass of target model
3. Implement rejection_sample() — the core acceptance/rejection logic
4. Implement the full SD loop and compare output to pure AR
5. Measure speedup vs AR for different K values
6. BONUS: Verify that SD produces the EXACT same distribution as AR

REFERENCES:
- Leviathan et al. 2023: "Fast Inference from Transformers via Speculative Decoding"
- Chen et al. 2023: "Accelerating LLM Decoding with Speculative Sampling"

Next: 02_rejection_sampling.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.mini_transformer import MiniTransformer

# ─────────────────────────────────────────────────────────────────────
# Step 1: Draft K tokens from the draft model
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def draft_tokens(draft_model, prefix, K, temperature=0.0):
    """
    Generate K draft tokens autoregressively from the draft model.
    
    Args:
        draft_model: The small/fast draft model
        prefix: [1, seq_len] tensor of token IDs
        K: Number of tokens to draft
        temperature: Sampling temperature (0 = greedy)
    
    Returns:
        draft_token_ids: [K] tensor of drafted token IDs
        draft_probs: [K, vocab_size] tensor of draft probabilities at each position
    
    TODO: Implement this!
    Hint: For each of K steps:
      1. Run draft model forward pass on current sequence
      2. Get logits at last position
      3. Compute probabilities (softmax with temperature)
      4. Sample a token
      5. Append to sequence
      6. Save the probability distribution (needed for rejection sampling)
    """
    generated = prefix.clone()
    draft_token_ids = []
    draft_probs_list = []

    for i in range(K):
        logits = draft_model(generated)
        next_logits = logits[:, -1, :] # (1, vocab_size)
        # probs = F.softmax(next_logits / max(temperature, 1e-8), dim=-1)  # (1, vocab_size)

        if temperature == 0:
            next_token = next_logits.argmax(dim=1, keepdim=True)
            probs = torch.zeros_like(next_logits)      # probs = [[0.0, 0.0, 0.0, 0.0, 0.0]]
            probs.scatter_(1, next_token, 1.0) #scatter_(dim, index, value), "along dimension 1, at the position given by next_token, write 1.0".
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        draft_token_ids.append(next_token.item())
        draft_probs_list.append(probs.squeeze(0)) # (vocab_size,)
        generated = torch.cat([generated, next_token], dim=1)
    
    return (
        torch.tensor(draft_token_ids, device=prefix.device), # [K]
        torch.stack(draft_probs_list)                        # [K, vocab_size]
    )


# ─────────────────────────────────────────────────────────────────────
# Step 2: Verify K tokens with the target model
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def verify_tokens(target_model, prefix, draft_token_ids, temperature=0.0):
    """
    Verify all K draft tokens in ONE forward pass of the target model.
    
    This is the key insight: we feed [prefix, d1, d2, ..., dK] to the target
    model and get logits at ALL positions simultaneously. The cost is roughly
    the same as generating a single token (memory-bandwidth bound).
    
    Args:
        target_model: The large/slow target model
        prefix: [1, seq_len] tensor
        draft_token_ids: [K] tensor of drafted tokens
        temperature: Sampling temperature
    
    Returns:
        target_probs: [K+1, vocab_size] tensor of target probabilities
            - target_probs[i] = p_target(· | prefix, d1, ..., d_i)
            - target_probs[K] is used for the bonus token if all accepted
    
    TODO: Implement this!
    Hint:
      1. Concatenate prefix + draft tokens
      2. Run ONE forward pass of target model
      3. Extract logits at positions [seq_len-1, seq_len, ..., seq_len+K-1]
         (these correspond to predicting d1, d2, ..., dK, and bonus)
      4. Convert to probabilities
    """
    # draft_token_ids is [K] (1D) but prefix is [1, seq_len] (2D)
    generated = torch.cat([prefix, draft_token_ids.unsqueeze(0)], dim=1)
    logits = target_model(generated)

    K = draft_token_ids.shape[0]
    seq_len = prefix.shape[1]
    verify_logits = logits[:, seq_len-1 : seq_len+K, :] # [1, K+1, vocab_size]
    temp = temperature if temperature > 0 else 1.0 # always want the full probability distribution regardless of temperature, because rejection sampling needs p_target for all tokens, not just the chosen one.
    return F.softmax(verify_logits / temp, dim=-1).squeeze(0)  # [K+1, vocab_size]


# ─────────────────────────────────────────────────────────────────────
# Step 3: Rejection Sampling
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def rejection_sample(draft_token_ids, draft_probs, target_probs):
    """
    The core of speculative decoding: decide which draft tokens to accept.
    
    For each drafted token d_i:
      - Accept with probability min(1, p_target(d_i) / p_draft(d_i))
      - On first rejection at position k:
        * Discard all tokens after position k
        * Sample a "bonus token" from the residual distribution:
          r(x) ∝ max(p_target(x) - p_draft(x), 0)
    
    If ALL tokens accepted:
      - Sample bonus token from p_target at position K
    
    This guarantees the output follows p_target EXACTLY (lossless).
    
    Args:
        draft_token_ids: [K] drafted tokens
        draft_probs: [K, vocab_size] draft probabilities
        target_probs: [K+1, vocab_size] target probabilities
    
    Returns:
        accepted_tokens: list of accepted token IDs (including bonus)
        num_accepted: number of draft tokens accepted (0 to K)
    
    TODO: Implement this!
    Hint:
      For i in range(K):
        1. r = uniform(0, 1)
        2. acceptance_prob = min(1, target_probs[i, d_i] / draft_probs[i, d_i])
        3. if r < acceptance_prob: accept d_i, continue
        4. else: reject d_i
           - Compute residual: res(x) = max(target_probs[i, x] - draft_probs[i, x], 0)
           - Normalize residual to a valid distribution
           - Sample bonus token from residual
           - Return accepted tokens + bonus token
      If all accepted:
        - Sample bonus from target_probs[K]
        - Return all K draft tokens + bonus
    """
    K = draft_token_ids.shape[0]
    accepted_tokens = []

    for i in range(K):
        d_i = draft_token_ids[i].item()
        p_target = target_probs[i, d_i].item()
        p_draft  = draft_probs[i, d_i].item()

        acceptance_prob = min(1.0, p_target / (p_draft + 1e-10))

        if torch.rand(1).item() < acceptance_prob: 
            # accept
            accepted_tokens.append(d_i)
        else:
            # reject — sample bonus from residual distribution
            residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
            residual = residual / (residual.sum() + 1e-10)  # normalize to a valid probability distribution again so sum up to 1
            bonus = torch.multinomial(residual, num_samples=1).item()
            accepted_tokens.append(bonus)
            return accepted_tokens, len(accepted_tokens) - 1  # -1 because bonus isn't a draft token
    
    # all K accepted — sample bonus from target at position K
    bonus = torch.multinomial(target_probs[K], num_samples=1).item()
    accepted_tokens.append(bonus)
    return accepted_tokens, K


# ─────────────────────────────────────────────────────────────────────
# Full Speculative Decoding Loop
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def speculative_decode(target_model, draft_model, prompt, max_new_tokens=64, 
                        K=5, temperature=0.0):
    """
    Full speculative decoding: draft K, verify, accept/reject, repeat.
    
    TODO: Implement using draft_tokens(), verify_tokens(), rejection_sample()
    """
    generated = prompt.clone()
    total_draft_tokens = 0
    total_accepted = 0
    
    while generated.shape[1] - prompt.shape[1] < max_new_tokens:
        # 1. Draft K tokens
        draft_ids, draft_probs = draft_tokens(draft_model, generated, K, temperature)
        total_draft_tokens += K
        
        # 2. Verify with target model
        target_probs = verify_tokens(target_model, generated, draft_ids, temperature)
        
        # 3. Rejection sampling
        accepted, num_accepted = rejection_sample(draft_ids, draft_probs, target_probs)
        total_accepted += num_accepted
        
        # 4. Append accepted tokens
        accepted_tensor = torch.tensor(accepted, device=generated.device).unsqueeze(0)
        generated = torch.cat([generated, accepted_tensor], dim=1)
    
    acceptance_rate = total_accepted / total_draft_tokens if total_draft_tokens > 0 else 0
    return generated, acceptance_rate


# ─────────────────────────────────────────────────────────────────────
# Main: Compare AR vs SD
# ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Target model: larger
    target = MiniTransformer(vocab_size=1000, hidden_size=512, num_layers=8, num_heads=4)
    target = target.to(device).eval()
    
    # Draft model: smaller (should be ~4-8x faster)
    draft = MiniTransformer(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4)
    draft = draft.to(device).eval()
    
    print(f"Target: {sum(p.numel() for p in target.parameters())/1e6:.1f}M params")
    print(f"Draft:  {sum(p.numel() for p in draft.parameters())/1e6:.1f}M params")
    
    prompt = torch.randint(0, 1000, (1, 16), device=device)

    # In main(), add this sanity check:
    print("\n--- Sanity check: same model as draft and target ---")
    same_model = MiniTransformer(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=4).to(device).eval()
    output, rate = speculative_decode(same_model, same_model, prompt, max_new_tokens=32, K=5, temperature=1.0)
    print(f"acceptance_rate with identical models: {rate:.2f}")  # should be ~1.0   
    
    # ── Compare AR vs SD for different K values ──
    print("\n" + "="*70)
    print("Speculative Decoding vs Autoregressive")
    print("="*70)
    
    for K in [1, 3, 5, 7, 10]:
        start = time.perf_counter()
        output, acceptance_rate = speculative_decode(
            target, draft, prompt, max_new_tokens=64, K=K
        )
        torch.cuda.synchronize()
        sd_time = time.perf_counter() - start
        
        print(f"  K={K:2d}: acceptance_rate={acceptance_rate:.2f}, "
              f"time={sd_time*1000:.0f}ms, "
              f"tokens/round={acceptance_rate*K+1:.1f}")
    
    print("""
    KEY INSIGHT:
    - Neither draft or target model is trained and both has random weights, so they output essentially random logits.
    - Higher K = more tokens per round IF acceptance rate stays high
    - But acceptance rate drops with larger K (harder to predict far ahead)
    - Sweet spot is usually K=3-7 depending on draft model quality
    
    → Next: 02_rejection_sampling.py (deep dive into the math)
    """)


if __name__ == "__main__":
    main()

"""
python -m llm.spec_decode.01_spec_decode_from_scratch
Target: 26.5M params
Draft:  0.7M params

--- Sanity check: same model as draft and target ---
acceptance_rate with identical models: 1.00

======================================================================
Speculative Decoding vs Autoregressive
======================================================================
  K= 1: acceptance_rate=0.00, time=300ms, tokens/round=1.0
  K= 3: acceptance_rate=0.00, time=384ms, tokens/round=1.0
  K= 5: acceptance_rate=0.00, time=492ms, tokens/round=1.0
  K= 7: acceptance_rate=0.00, time=605ms, tokens/round=1.0
  K=10: acceptance_rate=0.00, time=762ms, tokens/round=1.0

    KEY INSIGHT:
    - Neither draft or target model is trained and both has random weights, so they output essentially random logits.
    - Higher K = more tokens per round IF acceptance rate stays high
    - But acceptance rate drops with larger K (harder to predict far ahead)
    - Sweet spot is usually K=3-7 depending on draft model quality
"""