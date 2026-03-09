"""
00 — Autoregressive Decoding: The Bottleneck We're Solving
==========================================================

Before understanding speculative decoding, you need to deeply feel WHY
autoregressive (AR) decoding is slow. This script makes it visceral.

THE FUNDAMENTAL PROBLEM:
========================

    ┌──────────────────────────────────────────────────────────────────┐
    │ Autoregressive Decoding (one token at a time)                   │
    │                                                                  │
    │  Step 1: Load ALL model weights + KV cache → Generate token 1   │
    │  Step 2: Load ALL model weights + KV cache → Generate token 2   │
    │  Step 3: Load ALL model weights + KV cache → Generate token 3   │
    │  ...                                                             │
    │  Step N: Load ALL model weights + KV cache → Generate token N   │
    │                                                                  │
    │  Each step: ~10-30ms for a 70B model                             │
    │  For 512 tokens: 512 × 20ms = 10.24 seconds                     │
    │                                                                  │
    │  The GPU is mostly WAITING for data to arrive from memory!       │
    └──────────────────────────────────────────────────────────────────┘

    WHY is it memory-bound?
    
    For a 70B model in FP16:
    - Model weights: 140 GB
    - H100 memory bandwidth: 3.35 TB/s
    - Time to load all weights: 140 GB / 3.35 TB/s = 41.8 ms
    - Actual compute per token: ~0.14 TFLOPS (tiny!)
    - H100 peak compute: 989 TFLOPS
    - Compute utilization: 0.14 / 989 = 0.014% ← CRIMINALLY LOW
    
    The GPU's 989 TFLOPS of compute sits 99.98% idle during decode!

EXERCISES:
=========
1. Run this script and observe the per-token latency
2. Change the model size and see how latency scales with parameters
3. Calculate the theoretical max tokens/sec from memory bandwidth
4. Understand: speculative decoding turns N serial loads into 1 batched load

Next: 01_spec_decode_from_scratch.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# A tiny transformer for experimentation
# ─────────────────────────────────────────────────────────────────────
class TinyTransformer(nn.Module):
    """A minimal GPT-style model. Small enough to run on any GPU."""
    
    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=2, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(2048, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, 
            dim_feedforward=hidden_size * 4, batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size
        
    def forward(self, input_ids, positions=None):
        """Forward pass. Returns logits for next token prediction."""
        B, T = input_ids.shape
        if positions is None:
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        x = self.embed(input_ids) + self.pos_embed(positions)
        
        # Causal mask: each position can only attend to previous positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        logits = self.lm_head(x)
        return logits


# ─────────────────────────────────────────────────────────────────────
# Autoregressive Generation
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def autoregressive_generate(model, prompt_ids, max_new_tokens=64, temperature=0.0):
    """
    Standard autoregressive generation. One token at a time.
    
    This is what we want to speed up with speculative decoding.
    
    The key observation: each step does a FULL forward pass through the model,
    but only uses the LAST token's logits. This is extremely wasteful!
    
    In a production system, we'd use KV-cache to avoid recomputing attention
    for previous tokens. But even with KV-cache, we still need to:
    1. Load all model weights from HBM for every single token
    2. Compute attention over the growing KV-cache
    
    Both are memory-bandwidth-bound operations.
    """
    device = prompt_ids.device
    generated = prompt_ids.clone()
    
    per_token_times = []
    
    for i in range(max_new_tokens):
        start = time.perf_counter()
        
        # Full forward pass — in production, KV-cache would make this faster
        # but it's still one-token-at-a-time
        logits = model(generated)
        
        # Only use the last token's logits
        next_logits = logits[:, -1, :]
        
        # Sample next token
        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        per_token_times.append(elapsed)
    
    return generated, per_token_times


# ─────────────────────────────────────────────────────────────────────
# Main: Profile AR decoding
# ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ── Experiment 1: Basic AR profiling ──
    print("\n" + "="*70)
    print("EXPERIMENT 1: Autoregressive Decoding Profile")
    print("="*70)
    
    model = TinyTransformer(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=4)
    model = model.to(device).eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    param_bytes = num_params * 2  # FP16 = 2 bytes
    print(f"Model: {num_params:,} parameters ({param_bytes/1e6:.1f} MB in FP16)")
    
    prompt = torch.randint(0, 1000, (1, 16), device=device)
    
    # Warmup
    _ = autoregressive_generate(model, prompt, max_new_tokens=8)
    
    # Profile
    _, times = autoregressive_generate(model, prompt, max_new_tokens=64)
    
    avg_time = sum(times[5:]) / len(times[5:])  # skip first few (warmup)
    tokens_per_sec = 1.0 / avg_time
    
    print(f"\nPer-token latency: {avg_time*1000:.2f} ms")
    print(f"Tokens/sec: {tokens_per_sec:.1f}")
    print(f"For 512 tokens: {512 * avg_time:.2f} seconds")
    
    # ── Experiment 2: Scaling with model size ──
    print("\n" + "="*70)
    print("EXPERIMENT 2: How Latency Scales with Model Size")
    print("="*70)
    
    for hidden_size, num_layers in [(128, 2), (256, 4), (512, 8), (1024, 8)]:
        model = TinyTransformer(
            vocab_size=1000, hidden_size=hidden_size, 
            num_layers=num_layers, num_heads=4
        ).to(device).eval()
        
        num_params = sum(p.numel() for p in model.parameters())
        
        prompt = torch.randint(0, 1000, (1, 16), device=device)
        _ = autoregressive_generate(model, prompt, max_new_tokens=4)  # warmup
        _, times = autoregressive_generate(model, prompt, max_new_tokens=32)
        
        avg_time = sum(times[2:]) / len(times[2:])
        print(f"  hidden={hidden_size:4d}, layers={num_layers}, "
              f"params={num_params/1e6:.1f}M, "
              f"latency={avg_time*1000:.2f}ms, "
              f"tok/s={1/avg_time:.0f}")
    
    # ── Experiment 3: The batch=1 problem ──
    print("\n" + "="*70)
    print("EXPERIMENT 3: Batch Size 1 is the Worst Case")
    print("="*70)
    print("""
    At batch_size=1 during decode:
    
    - Each token requires loading ALL model weights from GPU memory
    - On H100: 140 GB weights / 3.35 TB/s bandwidth = 41.8 ms per token
    - That's only ~24 tokens/sec for a 70B model!
    - The 989 TFLOPS of compute is almost completely unused
    
    This is why speculative decoding helps:
    Instead of:  Load weights → 1 token → Load weights → 1 token → ...
    We do:       Load weights → verify K tokens at once → much better utilization!
    
    The key insight: verification of K tokens costs almost the same as
    generating 1 token, because the bottleneck is loading weights, not compute.
    """)
    
    # ── Key Takeaways ──
    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. AR decoding is SEQUENTIAL — each token depends on all previous tokens
    2. At batch_size=1, it's MEMORY-BANDWIDTH BOUND — GPU compute sits idle
    3. Per-token latency grows with model size (more weights to load)
    4. For 70B models: ~24 tokens/sec at BS=1 (theoretical max from bandwidth)
    
    SPECULATIVE DECODING solves this by:
    - Using a small, fast draft model to guess the next K tokens
    - Verifying all K tokens in ONE target model forward pass
    - Verification costs ~same as 1 token (memory-bound, not compute-bound)
    - If draft is good, we get K tokens for the price of ~1
    
    → Next: 01_spec_decode_from_scratch.py
    """)


if __name__ == "__main__":
    main()