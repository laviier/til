"""
03 — Tracing vLLM's Speculative Decoding Code
==============================================

This is a GUIDED READING of vLLM's speculative decoding implementation.
NOT a runnable script — it's a study guide with code pointers.

Read through this file, then go read the actual vLLM code at each pointer.
Set breakpoints and trace through with a debugger for maximum understanding.

HOW TO USE THIS:
===============
1. Read each section below
2. Open the referenced vLLM file
3. Find the referenced function/class
4. Read it with the context provided here
5. (Optional) Set a breakpoint and run vLLM with spec decode enabled

SETUP FOR TRACING:
=================
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --speculative-config '{"model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5}' \
        --enforce-eager  # disable compilation for easier debugging
"""

# ─────────────────────────────────────────────────────────────────────
# LAYER 1: Configuration
# File: vllm/config/speculative.py
# ─────────────────────────────────────────────────────────────────────
"""
START HERE. The SpeculativeConfig class controls everything.

Key fields to understand:
  - method: "eagle" | "eagle3" | "mtp" | "draft_model" | "ngram" | ...
  - num_speculative_tokens: K (how many tokens to draft)
  - parallel_drafting: bool (one-pass vs sequential drafting)
  - speculative_token_tree: defines tree structure for tree-based SD
  - draft_model_config: ModelConfig for the draft model

EXERCISE: Read __post_init__() and trace how 'method' is auto-detected.
"""

# ─────────────────────────────────────────────────────────────────────
# LAYER 2: The Proposer (Scheduler-side)
# File: vllm/v1/spec_decode/eagle.py
# ─────────────────────────────────────────────────────────────────────
"""
SpecDecodeBaseProposer is the main class. It:
  1. Holds the draft model
  2. Prepares inputs for drafting
  3. Runs K forward passes (or 1 if parallel_drafting)
  4. Returns draft token IDs

KEY METHOD: propose()
  - Takes target_hidden_states, next_token_ids, common_attn_metadata
  - Calls set_inputs_first_pass() to prepare inputs
  - Runs the draft model forward pass
  - Samples draft tokens via _greedy_sample()
  
  For num_speculative_tokens > 1 and NOT parallel_drafting:
    - Enters a loop: for token_index in range(num_speculative_tokens - 1)
    - Each iteration: update positions, seq_lens, slot_mapping, run model
    - This is the SEQUENTIAL dependency that SSD aims to eliminate!

EXERCISE: Read propose() and trace the sequential drafting loop.
Find the line "if self.num_speculative_tokens == 1 or self.parallel_drafting:"
This is where parallel drafting short-circuits the loop.
"""

# ─────────────────────────────────────────────────────────────────────
# LAYER 3: Parallel Drafting
# File: vllm/v1/spec_decode/eagle.py, set_inputs_first_pass()
# ─────────────────────────────────────────────────────────────────────
"""
When parallel_drafting=True, ALL K tokens are generated in ONE forward pass.

How it works:
  1. The input is expanded: for each request, add K extra "masked" positions
  2. Masked positions use a special token (pard_token / ptd_token_id)
  3. The attention mask allows each position to see the verified prefix
     but NOT other draft positions (causal mask with masking)
  4. One forward pass generates K tokens simultaneously

Key code:
  copy_and_expand_eagle_inputs_kernel (Triton kernel in vllm/v1/spec_decode/utils.py)
  - Copies target token IDs into expanded buffer
  - Fills extra slots with parallel_drafting_token_id
  - Sets is_masked_token_mask for the parallel positions

EXERCISE: 
  1. Read set_inputs_first_pass() in the `else` branch (needs_extra_input_slots)
  2. Understand how the Triton kernel expands the input
  3. Key question: what does the attention mask look like for parallel drafting?

THIS IS THE FOUNDATION FOR SSD:
  SSD extends parallel drafting to draft for MULTIPLE verification outcomes.
  Instead of K parallel positions for one outcome, SSD has B*F*(K+1) 
  positions covering all predicted outcomes.
"""

# ─────────────────────────────────────────────────────────────────────
# LAYER 4: The Speculator (Worker-side, MRV2)
# File: vllm/v1/worker/gpu/spec_decode/eagle/speculator.py
# ─────────────────────────────────────────────────────────────────────
"""
EagleSpeculator runs on the GPU worker. This is the MRV2 code path.

Key method: propose()
  1. Prepares eagle inputs (prepare_eagle_inputs - Triton kernel)
  2. Runs first forward pass (prefill-like, with full query)
  3. Samples first draft token
  4. Prepares decode inputs (prepare_eagle_decode - Triton kernel)
  5. Calls generate_draft() for remaining K-1 tokens
  
generate_draft() is the inner loop:
  For step in range(1, num_speculative_steps):
    1. run_model() - forward pass
    2. gumbel_sample() - sample draft token
    3. update_eagle_inputs() - prepare next step
    
This is where CUDA graphs are captured (capture_model).

EXERCISE: Read generate_draft() and understand the Triton kernels
that update inputs between steps. This is where SSD would insert
the speculation cache lookup.
"""

# ─────────────────────────────────────────────────────────────────────
# LAYER 5: Rejection Sampling
# File: vllm/v1/worker/gpu/spec_decode/rejection_sample.py
# ─────────────────────────────────────────────────────────────────────
"""
EXERCISE: Read the rejection sampling implementation.
Key questions:
  1. How does it handle the bonus token?
  2. How does it handle greedy vs random sampling?
  3. What information does it return that SSD could use for prediction?
"""

# ─────────────────────────────────────────────────────────────────────
# LAYER 6: The Engine Loop
# File: vllm/v1/engine/core.py
# ─────────────────────────────────────────────────────────────────────
"""
The engine core orchestrates the draft-verify loop.

Search for "spec_decode" in this file to find:
  1. How draft token IDs are passed to the scheduler
  2. How the scheduler coordinates draft and verify steps
  3. The async scheduling interaction with spec decode

KEY FOR SSD: The engine loop is where we'd insert the async overlap.
Instead of: scheduler → draft → verify → process
We'd have:  scheduler → verify+draft_async → process (if cache hit) or draft_sync
"""

# ─────────────────────────────────────────────────────────────────────
# SUMMARY: What to focus on for SSD
# ─────────────────────────────────────────────────────────────────────
"""
For SSD implementation, you need deep understanding of:

1. SpecDecodeBaseProposer.propose() — the main drafting entry point
2. The parallel_drafting code path — SSD extends this to multi-outcome
3. EagleSpeculator.propose() — the MRV2 worker-side code
4. Rejection sampling — to understand what SSD needs to predict
5. The engine core loop — to understand where async overlap is inserted
6. NCCL communication in vllm/v1/executor/ — for disaggregation

NEXT STEPS:
  04_draft_models.py — understand different draft model architectures
  05_parallel_drafting.py — implement parallel drafting from scratch
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nThis is a study guide, not a runnable script.")
    print("Open the referenced vLLM files and read along.")
