"""
02 — Rejection Sampling Deep Dive
==================================

The math behind speculative decoding's lossless guarantee.
This is the MOST important script for understanding SSD, because SSD's
speculation cache needs to predict the BONUS TOKEN — which comes from
the residual distribution studied here.

KEY CONCEPTS:
============
1. Acceptance probability: min(1, p_target(x) / p_draft(x))
2. Residual distribution: r(x) ∝ max(p_target(x) - p_draft(x), 0)
3. Acceptance rate: α = Σ min(p_target(x), p_draft(x))
4. Bonus token: sampled from residual (rejection) or target (all accepted)

WHY THIS MATTERS FOR SSD:
=========================
SSD needs to PREDICT the bonus token before verification completes.
The bonus token comes from the residual distribution, which depends on
both p_target and p_draft. SSD's key insight: the top-F draft logits
are good predictors of where the residual mass concentrates.

EXERCISES:
=========
1. Implement acceptance probability computation
2. Implement residual distribution computation  
3. Empirically verify the lossless property (histogram test)
4. Measure: what % of bonus tokens fall in top-F draft logits?
   (This directly measures SSD's cache hit rate!)
5. Implement Saguaro sampling: bias p_draft to control the residual

Next: 03_vllm_sd_tracing.py
"""

import torch
import torch.nn.functional as F
import collections


def compute_acceptance_rate(p_target, p_draft):
    """
    Compute the acceptance rate α = Σ_x min(p_target(x), p_draft(x))
    
    This equals 1 - 0.5 * ||p_target - p_draft||_1 (total variation distance)
    
    Args:
        p_target: [vocab_size] target distribution
        p_draft: [vocab_size] draft distribution
    Returns:
        alpha: scalar acceptance rate
    
    TODO: Implement this! (It's one line)
    """
    raise NotImplementedError("YOUR TURN")


def compute_residual_distribution(p_target, p_draft):
    """
    Compute the residual distribution: r(x) ∝ max(p_target(x) - p_draft(x), 0)
    
    The bonus token is sampled from this distribution when a draft token
    is rejected. Understanding this distribution is KEY for SSD.
    
    TODO: Implement! Hint: clamp, then normalize.
    """
    raise NotImplementedError("YOUR TURN")


def simulate_rejection_sampling(p_target, p_draft, num_samples=100000):
    """
    Simulate rejection sampling and verify it produces p_target exactly.
    
    Algorithm for ONE token:
    1. Sample x ~ p_draft
    2. Accept with prob min(1, p_target(x) / p_draft(x))
    3. If accepted: return x
    4. If rejected: sample from residual r(x) ∝ max(p_target(x) - p_draft(x), 0)
    
    TODO: Implement and collect histogram of outputs.
    Then compare histogram to p_target — they should match!
    """
    raise NotImplementedError("YOUR TURN")


def measure_bonus_token_predictability(p_target, p_draft, num_samples=10000):
    """
    THIS IS THE KEY MEASUREMENT FOR SSD.
    
    When a token is rejected, the bonus token comes from the residual.
    SSD predicts the bonus token using the top-F draft logits.
    
    Measure: for F=1,2,4,8,16,32, what fraction of bonus tokens
    fall in the top-F tokens of the draft distribution?
    
    This is exactly the "cache hit rate" from the SSD paper (Figure 3).
    
    TODO: 
    1. Compute residual distribution
    2. Sample many bonus tokens from residual
    3. For each F, check if bonus token is in top-F of p_draft
    4. Report hit rate for each F
    5. Fit power law: 1 - hit_rate(F) ~ F^(-r)
    """
    raise NotImplementedError("YOUR TURN")


def saguaro_sampling(draft_logits, F, C=0.5):
    """
    Saguaro sampling: modify draft distribution to increase cache hit rate.
    
    σ_{F,C}(z) ∝ C * exp(z_t)  if t in top_F(z)
                   exp(z_t)    otherwise
    
    By downweighting the top-F tokens in the draft distribution,
    the residual max(p_target - p_draft, 0) concentrates MORE mass
    on those same top-F tokens, making bonus tokens more predictable.
    
    Tradeoff: lower C → higher cache hit rate, but lower acceptance rate.
    
    TODO:
    1. Implement Saguaro sampling
    2. Measure acceptance rate vs cache hit rate for C = [0.0, 0.3, 0.5, 0.8, 1.0]
    3. Replicate Figure 5 from the SSD paper
    """
    raise NotImplementedError("YOUR TURN")


def main():
    torch.manual_seed(42)
    V = 100  # small vocab for visualization
    
    # Create synthetic target and draft distributions
    # Draft is a "blurred" version of target (simulating a weaker model)
    target_logits = torch.randn(V)
    draft_logits = target_logits + 0.5 * torch.randn(V)  # add noise
    
    p_target = F.softmax(target_logits, dim=0)
    p_draft = F.softmax(draft_logits, dim=0)
    
    print("="*60)
    print("EXERCISE 1: Acceptance Rate")
    print("="*60)
    alpha = compute_acceptance_rate(p_target, p_draft)
    print(f"  Acceptance rate α = {alpha:.4f}")
    print(f"  Expected tokens per round (K=5): {alpha*5+1:.2f}")
    
    print("\n" + "="*60)
    print("EXERCISE 2: Residual Distribution")
    print("="*60)
    residual = compute_residual_distribution(p_target, p_draft)
    print(f"  Residual has {(residual > 0.001).sum().item()} tokens with significant mass")
    
    print("\n" + "="*60)
    print("EXERCISE 3: Verify Lossless Property")
    print("="*60)
    # simulate_rejection_sampling should show output matches p_target
    
    print("\n" + "="*60)
    print("EXERCISE 4: Bonus Token Predictability (SSD Cache Hit Rate)")
    print("="*60)
    # measure_bonus_token_predictability shows how well top-F predicts bonus
    
    print("\n" + "="*60)
    print("EXERCISE 5: Saguaro Sampling Tradeoff")
    print("="*60)
    # saguaro_sampling shows the acceptance rate vs cache hit rate tradeoff


if __name__ == "__main__":
    main()
