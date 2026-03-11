"""
10 — Speculation Cache: The Core of SSD
========================================

THIS IS THE MOST IMPORTANT EXERCISE. It implements SSD's core data structure
and can be used directly as the POC validation script (Week 1, Days 2-3).

The speculation cache maps verification outcomes to pre-computed speculations:
    cache[(k_accepted, bonus_token)] = next_K_draft_tokens

If the actual verification outcome matches a cached entry → CACHE HIT
→ return pre-computed tokens instantly → zero draft latency!

This script:
1. Defines the cache data structure
2. Implements uniform and geometric fan-out allocation
3. Simulates cache hit rates with toy distributions
4. Implements Saguaro sampling integration
5. Computes theoretical SSD speedup from Theorem 7
6. Sweeps fan-out F and temperature to replicate paper's Figure 3

Run this: python -m llm.spec_decode.10_speculation_cache
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass


# =====================================================================
# 1. Core data structures
# =====================================================================
@dataclass
class VerificationOutcome:
    """Result of verifying K draft tokens."""
    k_accepted: int        # how many draft tokens were accepted (0 to K)
    bonus_token: int       # the bonus token sampled (from residual or target)


class SpeculationCache:
    """
    SSD's core data structure: pre-computed speculations for predicted outcomes.
    
    cache[(k, token)] = list of K draft token IDs for the next round
    """
    
    def __init__(self):
        self.cache: dict[tuple[int, int], list[int]] = {}
    
    def store(self, k_accepted: int, bonus_token: int, draft_tokens: list[int]):
        self.cache[(k_accepted, bonus_token)] = draft_tokens
    
    def lookup(self, outcome: VerificationOutcome) -> list[int] | None:
        key = (outcome.k_accepted, outcome.bonus_token)
        return self.cache.get(key, None)
    
    @property
    def size(self) -> int:
        return len(self.cache)
    
    def clear(self):
        self.cache.clear()


# =====================================================================
# 2. Fan-out allocation strategies
# =====================================================================
def uniform_fanout(budget: int, K: int) -> list[int]:
    """
    Uniform fan-out: allocate budget/(K+1) guesses at each acceptance position.
    
    Simple but suboptimal — doesn't account for the fact that early rejections
    are more likely than late ones.
    """
    per_position = max(1, budget // (K + 1))
    fanout = [per_position] * (K + 1)
    # Distribute remainder
    remainder = budget - sum(fanout)
    for i in range(remainder):
        fanout[i] += 1
    return fanout


def geometric_fanout(budget: int, K: int, acceptance_rate: float, r: float = 1.0) -> list[int]:
    """
    Geometric fan-out from Theorem 12 of the SSD paper.
    
    F_k = F_0 * a^(k/(1+r))  for k < K
    F_K = F_0 * a^(K/(1+r)) * (1-a)^(-1/(1+r))
    
    Intuition:
    - Early positions (k small) are more likely to have rejections
      (probability ~ (1-a) * a^k), so allocate MORE guesses there
    - Late positions (k large) are unlikely, so allocate FEWER
    - The last position (all accepted) has probability a^K, needs special handling
    
    Args:
        budget: total number of outcomes to prepare for
        K: speculation lookahead
        acceptance_rate: a (probability of accepting each token)
        r: power-law exponent for cache hit rate
    """
    a = acceptance_rate
    if a <= 0 or a >= 1:
        return uniform_fanout(budget, K)
    
    # Compute relative fan-out values (un-normalized)
    raw_fanout = []
    for k in range(K):
        raw_fanout.append(a ** (k / (1 + r)))
    # Last position: all accepted
    raw_fanout.append(a ** (K / (1 + r)) * (1 - a) ** (-1 / (1 + r)))
    
    # Normalize to budget
    total_raw = sum(raw_fanout)
    fanout = [max(1, round(f * budget / total_raw)) for f in raw_fanout]
    
    # Adjust to exactly match budget
    while sum(fanout) > budget:
        max_idx = fanout.index(max(fanout))
        fanout[max_idx] -= 1
    while sum(fanout) < budget:
        min_idx = fanout.index(min(fanout))
        fanout[min_idx] += 1
    
    return fanout


# =====================================================================
# 3. Build the speculation cache
# =====================================================================
def build_cache(
    draft_logits_per_position: list[torch.Tensor],  # [K+1] x [vocab_size]
    fanout: list[int],                                # [K+1] fan-out per position
    K: int,
    excluded_tokens: list[int] | None = None,         # tokens sent for verification
) -> SpeculationCache:
    """
    Build the speculation cache by taking top-F tokens at each position.
    
    At position k, we predict the bonus token by looking at the draft model's
    top-F_k logits (EXCLUDING the drafted token that was sent for verification,
    since the bonus token is guaranteed not to be the drafted token).
    
    Args:
        draft_logits_per_position: draft logits at each acceptance position
        fanout: number of bonus token guesses per position
        K: speculation length
        excluded_tokens: the drafted tokens (to exclude from cache)
    """
    cache = SpeculationCache()
    
    for k in range(K + 1):
        logits = draft_logits_per_position[k]
        
        # Exclude the drafted token (it can't be the bonus token)
        if excluded_tokens is not None and k < len(excluded_tokens):
            logits = logits.clone()
            logits[excluded_tokens[k]] = float('-inf')
        
        # Take top-F_k tokens as bonus token guesses
        top_tokens = torch.topk(logits, min(fanout[k], logits.shape[0])).indices
        
        for token in top_tokens.tolist():
            # In real SSD, we'd actually run the draft model to generate
            # the next K tokens starting from this outcome.
            # Here we use placeholder tokens for simulation.
            draft_tokens = list(range(K))  # placeholder
            cache.store(k, token, draft_tokens)
    
    return cache


# =====================================================================
# 4. Simulate cache hit rate
# =====================================================================
def simulate_cache_hit_rate(
    p_target: torch.Tensor,       # [vocab_size]
    p_draft: torch.Tensor,        # [vocab_size]
    draft_logits: torch.Tensor,   # [vocab_size] (raw logits)
    K: int = 5,
    acceptance_rate: float = 0.8,
    fanout_strategy: str = "geometric",
    budget: int = 32,
    num_simulations: int = 5000,
    r: float = 1.0,
) -> float:
    """
    Simulate the SSD cache hit rate.
    
    For each simulation:
    1. Simulate a verification round (how many accepted, which bonus token)
    2. Check if the outcome was in the cache
    """
    # Compute fan-out allocation
    if fanout_strategy == "uniform":
        fanout = uniform_fanout(budget, K)
    else:
        fanout = geometric_fanout(budget, K, acceptance_rate, r)
    
    # Build cache (using draft logits for all positions as simplification)
    draft_logits_per_pos = [draft_logits] * (K + 1)
    cache = build_cache(draft_logits_per_pos, fanout, K)
    
    # Compute residual distribution
    residual = torch.clamp(p_target - p_draft, min=0.0)
    residual_sum = residual.sum()
    if residual_sum > 0:
        residual = residual / residual_sum
    else:
        residual = p_target  # if draft == target, no residual
    
    # Simulate
    hits = 0
    for _ in range(num_simulations):
        # Simulate acceptance: each token accepted with prob acceptance_rate
        k_accepted = 0
        for k in range(K):
            if torch.rand(1).item() < acceptance_rate:
                k_accepted += 1
            else:
                break
        
        # Sample bonus token
        if k_accepted == K:
            # All accepted: bonus from target
            bonus = torch.multinomial(p_target, 1).item()
        else:
            # Rejected at position k_accepted: bonus from residual
            bonus = torch.multinomial(residual, 1).item()
        
        outcome = VerificationOutcome(k_accepted, bonus)
        if cache.lookup(outcome) is not None:
            hits += 1
    
    return hits / num_simulations


# =====================================================================
# 5. Saguaro sampling integration
# =====================================================================
def saguaro_sample(draft_logits: torch.Tensor, F: int, C: float = 0.5) -> torch.Tensor:
    """
    Saguaro sampling: suppress top-F tokens by factor C in draft distribution.
    Returns modified draft probabilities.
    """
    top_f_idx = torch.topk(draft_logits, F).indices
    modified = draft_logits.clone()
    modified[top_f_idx] += math.log(max(C, 1e-10))
    return torch.nn.functional.softmax(modified, dim=0)


# =====================================================================
# 6. Theoretical speedup (Theorem 7)
# =====================================================================
def compute_speedup(
    p_hit: float,
    E_hit: float,       # expected tokens generated on cache hit
    E_miss: float,      # expected tokens generated on cache miss  
    T_p: float = 0.3,   # draft time relative to verify (< 1 for SSD)
    T_b: float = 0.0,   # backup speculator time (0 for fast fallback)
) -> float:
    """
    Theorem 7 from the SSD paper.
    
    speedup_SSD = (p_hit * E_hit + (1-p_hit) * E_miss) /
                  (p_hit * max(1, T_p) + (1-p_hit) * (1 + T_b))
    """
    numerator = p_hit * E_hit + (1 - p_hit) * E_miss
    denominator = p_hit * max(1.0, T_p) + (1 - p_hit) * (1 + T_b)
    return numerator / denominator


def compute_sd_speedup(E_tokens: float, T_draft: float) -> float:
    """Standard SD speedup = E_tokens / (1 + T_draft)."""
    return E_tokens / (1 + T_draft)


# =====================================================================
# Main: Full SSD simulation
# =====================================================================
def main():
    torch.manual_seed(42)
    V = 200  # vocab size (small for speed)
    K = 5    # speculation length
    
    # Create realistic-ish target and draft distributions
    # Good draft = small noise added to target
    target_logits = torch.randn(V) * 2.0  # sharper distribution
    draft_logits = target_logits + 0.3 * torch.randn(V)  # close to target
    
    p_target = F.softmax(target_logits, dim=0)
    p_draft = F.softmax(draft_logits, dim=0)
    
    acceptance_rate = torch.minimum(p_target, p_draft).sum().item()
    
    print("=" * 70)
    print("  SSD Speculation Cache — Full Simulation")
    print("=" * 70)
    print(f"\n  Vocab size: {V}, K: {K}")
    print(f"  Acceptance rate (alpha): {acceptance_rate:.3f}")
    print(f"  Expected tokens per SD round: {acceptance_rate * K + 1:.2f}")
    
    # ── Part 1: Fan-out comparison ──
    print("\n" + "=" * 70)
    print("  PART 1: Uniform vs Geometric Fan-Out")
    print("=" * 70)
    
    budget = 24
    uf = uniform_fanout(budget, K)
    gf = geometric_fanout(budget, K, acceptance_rate, r=1.0)
    
    print(f"\n  Budget: {budget} total outcomes to cache")
    print(f"  Uniform fan-out:   {uf}  (sum={sum(uf)})")
    print(f"  Geometric fan-out: {gf}  (sum={sum(gf)})")
    print(f"""
    Geometric allocates more to early positions (where rejections are likely)
    and less to late positions (which require high alpha^k to reach).
    """)
    
    # ── Part 2: Cache hit rate vs fan-out budget ──
    print("=" * 70)
    print("  PART 2: Cache Hit Rate vs Fan-Out Budget")
    print("=" * 70)
    
    print(f"\n  {'Budget':>8}  {'Uniform':>10}  {'Geometric':>10}")
    print("  " + "-" * 32)
    
    for budget in [4, 8, 16, 32, 64]:
        hr_uniform = simulate_cache_hit_rate(
            p_target, p_draft, draft_logits, K, acceptance_rate,
            "uniform", budget, num_simulations=3000
        )
        hr_geometric = simulate_cache_hit_rate(
            p_target, p_draft, draft_logits, K, acceptance_rate,
            "geometric", budget, num_simulations=3000
        )
        print(f"  {budget:>8}  {hr_uniform:>10.3f}  {hr_geometric:>10.3f}")
    
    print("""
    Cache hit rate increases with budget (more guesses = more likely to hit).
    Geometric fan-out should be slightly better than uniform.
    Paper reports ~85-90% at F=8 with greedy decoding on real models.
    """)
    
    # ── Part 3: Saguaro sampling effect ──
    print("=" * 70)
    print("  PART 3: Saguaro Sampling Tradeoff")
    print("=" * 70)
    
    budget = 16
    print(f"\n  {'C':>6}  {'Accept Rate':>12}  {'Hit Rate':>10}")
    print("  " + "-" * 32)
    
    for C in [1.0, 0.7, 0.5, 0.3, 0.1]:
        p_saguaro = saguaro_sample(draft_logits, F=8, C=C)
        alpha_s = torch.minimum(p_target, p_saguaro).sum().item()
        hr = simulate_cache_hit_rate(
            p_target, p_saguaro, draft_logits, K, alpha_s,
            "geometric", budget, num_simulations=3000
        )
        print(f"  {C:>6.1f}  {alpha_s:>12.3f}  {hr:>10.3f}")
    
    print("""
    As C decreases: acceptance rate drops, but cache hit rate increases.
    The optimal C balances these for maximum end-to-end speedup.
    """)
    
    # ── Part 4: Theoretical speedup ──
    print("=" * 70)
    print("  PART 4: Theoretical SSD Speedup (Theorem 7)")
    print("=" * 70)
    
    E_hit = acceptance_rate * K + 1   # expected tokens on cache hit
    E_miss = 1.0                      # just the bonus token on miss
    T_draft = 0.3                     # draft takes 30% of verify time
    
    sd_speedup = compute_sd_speedup(E_hit, T_draft)
    
    print(f"\n  Standard SD speedup: {sd_speedup:.2f}x")
    print(f"  (E_tokens={E_hit:.2f}, T_draft={T_draft})")
    
    print(f"\n  {'p_hit':>6}  {'SSD Speedup':>12}  {'vs SD':>8}")
    print("  " + "-" * 30)
    
    for p_hit in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        ssd_speed = compute_speedup(p_hit, E_hit, E_miss, T_p=T_draft, T_b=0.0)
        print(f"  {p_hit:>6.2f}  {ssd_speed:>12.2f}x  {ssd_speed/sd_speedup:>7.2f}x SD")
    
    print(f"""
    At p_hit=0.85, SSD gets ~{compute_speedup(0.85, E_hit, E_miss, T_draft, 0.0):.1f}x speedup
    vs SD's {sd_speedup:.1f}x → SSD/SD ratio = {compute_speedup(0.85, E_hit, E_miss, T_draft, 0.0)/sd_speedup:.2f}x

    The paper reports 1.5-1.6x over SD baselines at BS=1 greedy.
    """)
    
    # ── Part 5: Power-law fit ──
    print("=" * 70)
    print("  PART 5: Power-Law Cache Miss Rate")
    print("=" * 70)
    print("""
    The SSD paper (Definition 11) assumes cache miss rate follows a power law:
        1 - p_hit(F) = 1/F^r
    
    This means: rejection rate drops as a power of fan-out F.
    Fitting r from empirical data tells us how quickly hits improve with F.
    Higher r = cache gets effective faster with budget.
    """)
    
    # Measure rejection rates at different fan-outs
    print(f"\n  {'F':>4}  {'Hit Rate':>10}  {'1-Hit':>8}  {'log(1-Hit)':>12}  {'log(F)':>8}")
    print("  " + "-" * 48)
    
    log_f_vals, log_miss_vals = [], []
    for F_val in [2, 4, 8, 16, 32]:
        hr = simulate_cache_hit_rate(
            p_target, p_draft, draft_logits, K, acceptance_rate,
            "uniform", F_val * (K + 1), num_simulations=3000
        )
        miss = max(1 - hr, 1e-6)
        log_f = math.log(F_val)
        log_miss = math.log(miss)
        log_f_vals.append(log_f)
        log_miss_vals.append(log_miss)
        print(f"  {F_val:>4}  {hr:>10.3f}  {miss:>8.3f}  {log_miss:>12.3f}  {log_f:>8.3f}")
    
    # Simple linear regression: log(miss) = -r * log(F) + c
    if len(log_f_vals) >= 2:
        n = len(log_f_vals)
        sum_x = sum(log_f_vals)
        sum_y = sum(log_miss_vals)
        sum_xy = sum(x * y for x, y in zip(log_f_vals, log_miss_vals))
        sum_xx = sum(x * x for x in log_f_vals)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2 + 1e-10)
        r_fit = -slope
        print(f"\n  Fitted power-law exponent r = {r_fit:.2f}")
        print(f"  (Paper typically finds r ~ 0.5-1.5 for real models)")
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │ THIS IS YOUR POC VALIDATION SCRIPT!                         │
    │                                                             │
    │ To validate SSD on real models (Week 1, Days 2-3):          │
    │                                                             │
    │ 1. Replace toy distributions with real Llama-3.2-1B draft   │
    │    and Llama-3.1-70B target logits                          │
    │ 2. Run on HumanEval + GSM8k prompts                        │
    │ 3. Measure p_hit(F) for F=1..64                             │
    │ 4. Fit power-law exponent r                                 │
    │ 5. Compute theoretical speedup from Theorem 7               │
    │                                                             │
    │ GO if:  p_hit(F=8) >= 75% at greedy                        │
    │         Theoretical speedup >= 1.4x over SD                 │
    └─────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()