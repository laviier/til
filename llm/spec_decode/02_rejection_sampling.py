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

    Concrete example with vocab_size=4:
        p_target = [0.5, 0.3, 0.1, 0.1]
        p_draft  = [0.2, 0.4, 0.3, 0.1]

        torch.minimum(p_target, p_draft) = [0.2, 0.3, 0.1, 0.1]
                                             ↑    ↑    ↑    ↑
                                           min  min  min  min of each pair

        .sum() = 0.2 + 0.3 + 0.1 + 0.1 = 0.7
        So α = 0.7 means the two models agree 70% of the time.

    The intuition: for each token x, min(p_target, p_draft) is the probability mass both models "share" on that token. 
    If both assign high probability to the same tokens, the overlap is large → high acceptance rate. 
    If they disagree everywhere, the overlap is small → low acceptance rate.
    This connects directly to the rejection sampling math:
        acceptance_prob for token x = min(1, p_target(x) / p_draft(x))

        α = Σ_x p_draft(x) * min(1, p_target(x) / p_draft(x))
          = Σ_x min(p_draft(x), p_target(x))
    It's also equivalent to 1 - TV(p_target, p_draft) where TV is the total variation distance — a standard measure of how different two distributions are.

    Args:
        p_target: [vocab_size] target distribution
        p_draft:  [vocab_size] draft distribution
    Returns:
        alpha: scalar acceptance rate
    """
    return torch.minimum(p_target, p_draft).sum().item()


def compute_residual_distribution(p_target, p_draft):
    """
    Compute the residual distribution: r(x) ∝ max(p_target(x) - p_draft(x), 0)

    The bonus token is sampled from this distribution when a draft token
    is rejected. Understanding this distribution is KEY for SSD.
    """
    residual = torch.clamp(p_target - p_draft, min=0.0)
    return residual / (residual.sum() + 1e-10)


def simulate_rejection_sampling(p_target, p_draft, num_samples=100000):
    """
    Simulate rejection sampling and verify it produces p_target exactly.

    Algorithm for ONE token:
    1. Sample x ~ p_draft
    2. Accept with prob min(1, p_target(x) / p_draft(x))
    3. If accepted: return x
    4. If rejected: sample from residual r(x) ∝ max(p_target(x) - p_draft(x), 0)

    Returns empirical distribution — should match p_target closely.
    """
    residual = compute_residual_distribution(p_target, p_draft)
    counts = torch.zeros_like(p_target)

    for _ in range(num_samples):
        x = torch.multinomial(p_draft, num_samples=1).item()
        acceptance_prob = min(1.0, (p_target[x] / (p_draft[x] + 1e-10)).item())
        if torch.rand(1).item() < acceptance_prob:
            counts[x] += 1
        else:
            bonus = torch.multinomial(residual, num_samples=1).item()
            counts[bonus] += 1

    return counts / counts.sum()


def measure_bonus_token_predictability(p_target, p_draft, num_samples=10000):
    """
    THIS IS THE KEY MEASUREMENT FOR SSD.

    When a token is rejected, the bonus token comes from the residual.
    SSD predicts the bonus token using the top-F draft logits.

    Measure: for F=1,2,4,8,16,32, what fraction of bonus tokens
    fall in the top-F tokens of the draft distribution?

    This is exactly the "cache hit rate" from the SSD paper (Figure 3).
    """
    residual = compute_residual_distribution(p_target, p_draft)
    bonus_tokens = torch.multinomial(residual, num_samples=num_samples, replacement=True)

    hit_rates = {}
    for top_f in [1, 2, 4, 8, 16, 32]:
        top_indices = torch.topk(p_draft, top_f).indices
        hits = sum(1 for t in bonus_tokens if t in top_indices)
        hit_rates[top_f] = hits / num_samples
        print(f"  Top-{top_f:2d} hit rate: {hit_rates[top_f]:.3f}")

    return hit_rates


def saguaro_sampling(draft_logits, top_f, C=0.5):
    """
    Saguaro sampling: modify draft distribution to increase cache hit rate.

    σ_{F,C}(z) ∝ C * exp(z_t)  if t in top_F(z)
                   exp(z_t)    otherwise

    By downweighting the top-F tokens in the draft distribution,
    the residual max(p_target - p_draft, 0) concentrates MORE mass
    on those same top-F tokens, making bonus tokens more predictable.

    Tradeoff: lower C → higher cache hit rate, but lower acceptance rate.
    """
    top_f_indices = torch.topk(draft_logits, top_f).indices
    # scale mask: C for top-F tokens, 1.0 for the rest
    scale = torch.ones_like(draft_logits)
    scale[top_f_indices] = C
    # apply in log space: log(C * exp(z)) = z + log(C)
    modified_logits = draft_logits + torch.log(scale + 1e-10)
    return F.softmax(modified_logits, dim=0)


def main():
    torch.manual_seed(42)
    V = 100  # small vocab for visualization

    # Draft is a "blurred" version of target (simulating a weaker model)
    target_logits = torch.randn(V)
    draft_logits = target_logits + 0.5 * torch.randn(V)

    p_target = F.softmax(target_logits, dim=0)
    p_draft  = F.softmax(draft_logits, dim=0)

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
    empirical = simulate_rejection_sampling(p_target, p_draft, num_samples=100000)
    l1_error = (empirical - p_target).abs().sum().item()
    print(f"  L1 error between empirical and p_target: {l1_error:.4f}")
    print(f"  (Should be close to 0.0 — lossless guarantee)")

    print("\n" + "="*60)
    print("EXERCISE 4: Bonus Token Predictability (SSD Cache Hit Rate)")
    print("="*60)
    measure_bonus_token_predictability(p_target, p_draft)

    print("\n" + "="*60)
    print("EXERCISE 5: Saguaro Sampling Tradeoff")
    print("="*60)
    print(f"  {'C':>4}   {'accept_rate':>12}   {'top-8 hit rate':>14}")
    for C in [0.0, 0.3, 0.5, 0.8, 1.0]:
        saguaro_draft = saguaro_sampling(draft_logits, top_f=8, C=C)
        alpha_s = compute_acceptance_rate(p_target, saguaro_draft)
        residual_s = compute_residual_distribution(p_target, saguaro_draft)
        bonus_tokens = torch.multinomial(residual_s, num_samples=10000, replacement=True)
        top8 = torch.topk(saguaro_draft, 8).indices
        hit_rate = sum(1 for t in bonus_tokens if t in top8) / 10000
        print(f"  C={C:.1f}   α={alpha_s:.4f}         hit@8={hit_rate:.3f}")


if __name__ == "__main__":
    main()
