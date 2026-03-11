"""
02b — SSD Intuition Builder: Why Can We Predict the Bonus Token?
================================================================

You understood 01: speculative decoding drafts K tokens, verifies them,
and on rejection samples a "bonus token" from the residual distribution.

The BIG question SSD asks: can we PREDICT that bonus token BEFORE 
verification finishes? If yes, we can pre-compute the next draft 
while verification is happening → hide the drafting latency entirely.

This script builds the intuition through concrete numbers.
Run it section by section and THINK about each output before moving on.

LEARNING STRATEGY:
=================
Don't try to understand the math abstractly. Instead:
1. Look at the NUMBERS in each example
2. Ask yourself "why does this make sense?"  
3. Only then read the explanation

Run this: python -m llm.spec_decode.02b_ssd_intuition
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# =====================================================================
# PART 1: What does the bonus token look like?
# =====================================================================
def part1_what_is_bonus_token():
    section("PART 1: What IS the bonus token?")
    
    print("""
    Recap from exercise 01:
    - Draft model proposes token d_i
    - Target model either ACCEPTS or REJECTS it
    - On REJECTION: we sample a "bonus token" from the RESIDUAL distribution
    - On ALL ACCEPTED: we sample bonus from the TARGET distribution at pos K
    
    Let's see what happens with a concrete 5-token vocabulary.
    """)
    
    # A concrete example with just 5 tokens for clarity
    # Imagine: token 0="the", 1="cat", 2="dog", 3="sat", 4="ran"
    token_names = ["the", "cat", "dog", "sat", "ran"]
    
    p_target = torch.tensor([0.40, 0.30, 0.15, 0.10, 0.05])  # target thinks "the" is most likely
    p_draft  = torch.tensor([0.10, 0.25, 0.35, 0.20, 0.10])  # draft thinks "dog" is most likely
    
    print("  Target distribution p_target:")
    for i, name in enumerate(token_names):
        bar = "█" * int(p_target[i] * 50)
        print(f"    {name:>4}: {p_target[i]:.2f} {bar}")
    
    print("\n  Draft distribution p_draft:")
    for i, name in enumerate(token_names):
        bar = "█" * int(p_draft[i] * 50)
        print(f"    {name:>4}: {p_draft[i]:.2f} {bar}")
    
    # The residual is where the bonus token comes from
    residual_raw = torch.clamp(p_target - p_draft, min=0.0)
    residual = residual_raw / residual_raw.sum()
    
    print("\n  Residual r(x) = max(p_target(x) - p_draft(x), 0), normalized:")
    for i, name in enumerate(token_names):
        diff = (p_target[i] - p_draft[i]).item()
        bar = "█" * int(residual[i] * 50)
        print(f"    {name:>4}: p_t={p_target[i]:.2f} - p_d={p_draft[i]:.2f} = {diff:+.2f} → r={residual[i]:.2f} {bar}")
    
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │ KEY INSIGHT #1:                                         │
    │                                                         │
    │ The residual distribution has mass ONLY where the       │
    │ target assigns MORE probability than the draft.         │
    │                                                         │
    │ In this example: "the" gets 0.40 from target but only   │
    │ 0.10 from draft → residual concentrates on "the" (0.86) │
    │                                                         │
    │ The bonus token is almost certainly "the" — the token   │
    │ the target model likes MORE than the draft model.       │
    └─────────────────────────────────────────────────────────┘
    """)
    
    return p_target, p_draft, residual, token_names


# =====================================================================
# PART 2: Can we predict it from draft logits alone?
# =====================================================================
def part2_prediction_from_draft(p_target, p_draft, residual, token_names):
    section("PART 2: Can we predict the bonus token WITHOUT seeing p_target?")
    
    print("""
    SSD's problem: we need to predict the bonus token BEFORE verification.
    That means we DON'T have p_target yet — only p_draft.
    
    Naive approach: guess the top-F tokens from p_draft.
    Let's see if that works...
    """)
    
    # Sort draft tokens by draft probability
    draft_sorted_indices = torch.argsort(p_draft, descending=True)
    
    print("  Draft model's ranking (most likely → least likely):")
    for rank, idx in enumerate(draft_sorted_indices):
        print(f"    Rank {rank+1}: '{token_names[idx]}' (p_draft={p_draft[idx]:.2f}, "
              f"residual={residual[idx]:.2f})")
    
    print("\n  Now let's check: if we cache the top-F draft tokens, what's the hit rate?")
    
    for F in [1, 2, 3, 4]:
        top_f_indices = torch.topk(p_draft, F).indices
        top_f_names = [token_names[i] for i in top_f_indices]
        # What fraction of the residual mass falls on these top-F tokens?
        hit_mass = residual[top_f_indices].sum().item()
        print(f"    Top-{F} draft tokens: {top_f_names}")
        print(f"      Residual mass covered: {hit_mass:.2f} ({hit_mass*100:.0f}% cache hit rate)")
        print()
    
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │ OBSERVATION:                                            │
    │                                                         │
    │ Top-1 of draft ("dog") has 0% hit rate!                │
    │ The draft's FAVORITE token has ZERO residual mass.      │
    │                                                         │
    │ Why? Because if the draft strongly favors "dog",        │
    │ p_draft("dog") > p_target("dog"), so the residual      │
    │ max(p_target - p_draft, 0) is ZERO for "dog".           │
    │                                                         │
    │ The residual concentrates on tokens the draft            │
    │ UNDERESTIMATES — tokens with p_target >> p_draft.       │
    └─────────────────────────────────────────────────────────┘
    """)
    
    print("""
    Wait — so the TOP draft tokens are BAD predictors of the bonus token?
    Not exactly. The SSD paper found that empirically, with REAL models 
    (not our toy example), the top draft logits ARE good predictors.
    
    Why? Because real draft models are trained to APPROXIMATE the target.
    When they're close, the residual is small and tends to concentrate
    on the same high-probability tokens.
    
    Let's see this with a BETTER draft model...
    """)


# =====================================================================
# PART 3: Good draft model vs bad draft model
# =====================================================================
def part3_draft_quality_matters():
    section("PART 3: Draft Quality Determines Predictability")
    
    token_names = ["the", "cat", "dog", "sat", "ran"]
    p_target = torch.tensor([0.40, 0.30, 0.15, 0.10, 0.05])
    
    print("  Target: ", {n: f"{p:.2f}" for n, p in zip(token_names, p_target.tolist())})
    print()
    
    # Bad draft: very different from target
    p_bad_draft = torch.tensor([0.10, 0.25, 0.35, 0.20, 0.10])
    res_bad = torch.clamp(p_target - p_bad_draft, min=0.0)
    res_bad = res_bad / (res_bad.sum() + 1e-10)
    
    # Good draft: close to target
    p_good_draft = torch.tensor([0.35, 0.28, 0.18, 0.12, 0.07])
    res_good = torch.clamp(p_target - p_good_draft, min=0.0)
    res_good = res_good / (res_good.sum() + 1e-10)
    
    print("  BAD draft (very different from target):")
    print(f"    Draft:    {dict(zip(token_names, [f'{p:.2f}' for p in p_bad_draft.tolist()]))}")
    print(f"    Residual: {dict(zip(token_names, [f'{r:.2f}' for r in res_bad.tolist()]))}")
    top2_bad = torch.topk(p_bad_draft, 2).indices
    hit_bad = res_bad[top2_bad].sum().item()
    print(f"    Top-2 draft tokens cover {hit_bad*100:.0f}% of residual mass")
    alpha_bad = torch.minimum(p_target, p_bad_draft).sum().item()
    print(f"    Acceptance rate: {alpha_bad:.2f}")
    
    print()
    print("  GOOD draft (close to target):")
    print(f"    Draft:    {dict(zip(token_names, [f'{p:.2f}' for p in p_good_draft.tolist()]))}")
    print(f"    Residual: {dict(zip(token_names, [f'{r:.2f}' for r in res_good.tolist()]))}")
    top2_good = torch.topk(p_good_draft, 2).indices
    hit_good = res_good[top2_good].sum().item()
    print(f"    Top-2 draft tokens cover {hit_good*100:.0f}% of residual mass")
    alpha_good = torch.minimum(p_target, p_good_draft).sum().item()
    print(f"    Acceptance rate: {alpha_good:.2f}")
    
    print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │ KEY INSIGHT #2:                                         │
    │                                                         │
    │ When draft ≈ target (good draft model), the residual    │
    │ is SMALL and CONCENTRATED on a few tokens.              │
    │                                                         │
    │ These few tokens tend to overlap with the draft's       │
    │ top tokens → high cache hit rate!                       │
    │                                                         │
    │ Bad draft: top-2 covers {hit_bad*100:.0f}% of residual             │
    │ Good draft: top-2 covers {hit_good*100:.0f}% of residual           │
    │                                                         │
    │ This is why SSD works in practice — real EAGLE/1B       │
    │ draft models are good enough that top-8 covers 85-90%   │
    │ of the residual mass.                                   │
    └─────────────────────────────────────────────────────────┘
    """)


# =====================================================================
# PART 4: What Saguaro Sampling does (the clever trick)
# =====================================================================
def part4_saguaro_trick():
    section("PART 4: Saguaro Sampling — The Clever Trick")
    
    print("""
    Problem: even with a good draft model, the top-F draft tokens 
    don't always cover the residual well.
    
    Saguaro's trick: DELIBERATELY make the draft distribution WORSE
    on the top-F tokens, so the residual MUST concentrate there.
    
    It's like poker: intentionally "under-bidding" on your best cards
    so that when you lose (rejection), you lose in a predictable way.
    """)
    
    token_names = ["the", "cat", "dog", "sat", "ran"]
    p_target = torch.tensor([0.40, 0.30, 0.15, 0.10, 0.05])
    
    # Original draft logits (before softmax)
    draft_logits = torch.tensor([1.5, 1.2, 0.5, 0.2, -0.5])
    p_draft = F.softmax(draft_logits, dim=0)
    
    print("  BEFORE Saguaro sampling:")
    print(f"    Draft logits:  {dict(zip(token_names, [f'{l:.1f}' for l in draft_logits.tolist()]))}")
    print(f"    Draft probs:   {dict(zip(token_names, [f'{p:.3f}' for p in p_draft.tolist()]))}")
    
    res_before = torch.clamp(p_target - p_draft, min=0.0)
    res_before = res_before / (res_before.sum() + 1e-10)
    print(f"    Residual:      {dict(zip(token_names, [f'{r:.3f}' for r in res_before.tolist()]))}")
    
    top2 = torch.topk(p_draft, 2).indices
    hit_before = res_before[top2].sum().item()
    alpha_before = torch.minimum(p_target, p_draft).sum().item()
    print(f"    Top-2 covers {hit_before*100:.1f}% of residual, acceptance rate = {alpha_before:.3f}")
    
    print(f"\n  Now apply Saguaro with C=0.3 (suppress top-2 tokens):")
    
    # Saguaro: multiply top-F logits by C in probability space
    # In log space: add log(C) to top-F logits
    C = 0.3
    F_val = 2
    top_f_idx = torch.topk(draft_logits, F_val).indices
    
    modified_logits = draft_logits.clone()
    modified_logits[top_f_idx] += torch.log(torch.tensor(C))  # equivalent to multiplying by C
    p_saguaro = F.softmax(modified_logits, dim=0)
    
    print(f"    Modified logits: {dict(zip(token_names, [f'{l:.1f}' for l in modified_logits.tolist()]))}")
    print(f"    Saguaro probs:   {dict(zip(token_names, [f'{p:.3f}' for p in p_saguaro.tolist()]))}")
    
    res_after = torch.clamp(p_target - p_saguaro, min=0.0)
    res_after = res_after / (res_after.sum() + 1e-10)
    print(f"    New residual:    {dict(zip(token_names, [f'{r:.3f}' for r in res_after.tolist()]))}")
    
    top2_saguaro = torch.topk(p_saguaro, F_val).indices
    hit_after = res_after[top2_saguaro].sum().item()
    alpha_after = torch.minimum(p_target, p_saguaro).sum().item()
    print(f"    Top-2 covers {hit_after*100:.1f}% of residual, acceptance rate = {alpha_after:.3f}")
    
    print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │ WHAT HAPPENED:                                          │
    │                                                         │
    │ By SUPPRESSING "the" and "cat" in the draft (C=0.3),   │
    │ their draft probabilities DECREASED.                    │
    │                                                         │
    │ Since residual = max(p_target - p_draft, 0):            │
    │ - Lower p_draft for "the" → BIGGER gap → MORE residual  │
    │ - The residual now concentrates on our cached tokens!    │
    │                                                         │
    │ Cache hit:  {hit_before*100:.0f}% → {hit_after*100:.0f}% (better prediction!)              │
    │ Accept rate: {alpha_before:.3f} → {alpha_after:.3f} (slightly worse)          │
    │                                                         │
    │ This is THE tradeoff: C controls the balance between    │
    │ cache hit rate and acceptance rate.                      │
    └─────────────────────────────────────────────────────────┘
    """)
    
    # Show the full tradeoff curve
    print("  Full C sweep (F=2):")
    print(f"    {'C':>4}  {'Accept Rate':>12}  {'Hit Rate':>10}  {'Net Benefit':>12}")
    for C_val in [1.0, 0.8, 0.5, 0.3, 0.1, 0.01]:
        mod_logits = draft_logits.clone()
        mod_logits[top_f_idx] += torch.log(torch.tensor(max(C_val, 1e-10)))
        p_s = F.softmax(mod_logits, dim=0)
        res_s = torch.clamp(p_target - p_s, min=0.0)
        res_s = res_s / (res_s.sum() + 1e-10)
        top2_s = torch.topk(p_s, F_val).indices
        hit = res_s[top2_s].sum().item()
        alpha = torch.minimum(p_target, p_s).sum().item()
        # Net benefit approximation: more hits save draft time, lower alpha costs tokens
        print(f"    {C_val:>4.2f}  {alpha:>12.3f}  {hit:>10.3f}  {'↑ better' if hit > 0.8 and alpha > 0.7 else ''}")


# =====================================================================
# PART 5: Putting it all together — the SSD loop
# =====================================================================
def part5_ssd_full_picture():
    section("PART 5: The Full SSD Picture")
    
    print("""
    Now you understand the pieces. Here's how SSD works end-to-end:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  REGULAR SD:                                                │
    │  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐      │
    │  │ Draft  │───→│ Verify │───→│ Draft  │───→│ Verify │      │
    │  │ (5ms)  │    │ (20ms) │    │ (5ms)  │    │ (20ms) │      │
    │  └────────┘    └────────┘    └────────┘    └────────┘      │
    │  Total: 25ms per round                                      │
    │                                                             │
    │  SSD (with cache hit):                                      │
    │  ┌────────────────────────────────────────────────┐         │
    │  │ Verify (20ms)                                  │         │
    │  │                                                │         │
    │  │  Meanwhile, draft pre-computes speculations    │         │
    │  │  for top-F verification outcomes...            │         │
    │  └────────────────────────────────────────────────┘         │
    │  │ Cache lookup: 0ms (instant!)                   │         │
    │  ┌────────────────────────────────────────────────┐         │
    │  │ Verify (20ms)                                  │         │
    │  └────────────────────────────────────────────────┘         │
    │  Total: 20ms per round (saved 5ms = the draft time!)        │
    │                                                             │
    │  SSD (with cache miss):                                     │
    │  ┌────────────────────────────────────────────────┐         │
    │  │ Verify (20ms)                                  │         │
    │  │  [pre-computed speculations DON'T match]       │         │
    │  └────────────────────────────────────────────────┘         │
    │  ┌─────────┐                                                │
    │  │ Fallback│  Draft from scratch (5ms, or use n-gram 0ms)  │
    │  └─────────┘                                                │
    │  Total: 25ms (same as regular SD — no worse!)               │
    └─────────────────────────────────────────────────────────────┘
    
    The "verification outcome" is: (how many tokens accepted, bonus token)
    
    Example with K=5:
      Possible outcomes:  (0, "the"), (0, "cat"), ...,
                          (1, "dog"), (1, "sat"), ...,
                          (2, "ran"), ...,
                          (5, "the"), (5, "cat"), ...  ← all 5 accepted
    
    With F=8 fan-out at each position: 6 positions × 8 guesses = 48 outcomes
    But we can't afford 48 separate draft runs! So we use a BUDGET B.
    
    The GEOMETRIC FAN-OUT (Theorem 12) says:
      - Position 0 (0 accepted = first token rejected): high probability,
        so allocate MORE bonus token guesses here
      - Position 5 (all accepted): lower probability (α^5),
        so allocate FEWER guesses
      - The allocation follows: F_k = F_0 * α^(k/(1+r))
        which is a geometric series — more guesses early, fewer late
    
    And SAGUARO SAMPLING makes each guess more likely to be correct
    by biasing the draft distribution.
    """)

    print("""
    ┌─────────────────────────────────────────────────────────┐
    │ SUMMARY — The 3 SSD Innovations:                        │
    │                                                         │
    │ 1. SPECULATION CACHE: Pre-compute drafts for multiple   │
    │    possible verification outcomes (the "what if" cache)  │
    │                                                         │
    │ 2. GEOMETRIC FAN-OUT: Allocate budget wisely — more     │
    │    guesses where rejections are likely (early positions) │
    │                                                         │
    │ 3. SAGUARO SAMPLING: Bias the draft distribution to     │
    │    make the bonus token land on cached tokens            │
    │                                                         │
    │ Together: ~85-90% cache hit rate at greedy decoding      │
    │ → draft latency hidden ~85-90% of the time              │
    │ → ~1.5x speedup over regular SD at batch size 1         │
    └─────────────────────────────────────────────────────────┘
    """)


def main():
    p_target, p_draft, residual, token_names = part1_what_is_bonus_token()
    
    input("\n  [Press Enter to continue to Part 2...]\n")
    part2_prediction_from_draft(p_target, p_draft, residual, token_names)
    
    input("\n  [Press Enter to continue to Part 3...]\n")
    part3_draft_quality_matters()
    
    input("\n  [Press Enter to continue to Part 4...]\n")
    part4_saguaro_trick()
    
    input("\n  [Press Enter to continue to Part 5...]\n")
    part5_ssd_full_picture()
    
    print("\n  ✅ You now understand the core intuition behind SSD!")
    print("  Next steps:")
    print("  - Go back to 02_rejection_sampling.py and re-implement from scratch")
    print("  - Focus on measure_bonus_token_predictability() — can you predict")
    print("    the numbers before running the code?")
    print("  - Then move to 03_vllm_sd_tracing.py")


if __name__ == "__main__":
    main()