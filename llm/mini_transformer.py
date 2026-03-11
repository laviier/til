import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniAttentionBlock(nn.Module):
    """
    Mini multi-head self-attention block with pre-norm and residual connection to attend at other token

    Applies causal masking so each token can only attend to itself and previous tokens.
    Compute: LayerNorm -> QKV projections -> scaled dot-product attention -> out projection -> Residual Add.
    """

    def __init__(self, hidden_size: int = 1024, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads # divid 1024 features into 8 heads so each owns 128 features to form their perspective
        self.norm = nn.LayerNorm(hidden_size) # normalize the 1024 numbers for each token so they have mean ≈ 0 and variance ≈ 1
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False) # (B, T, 1024) × (1024, 1024) → (B, T, 1024), bias=False means no bias term is added
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False) # mixing step to combine output of all 8 heads
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        residual = x # always save original x so we don't lose information from earlier layers
        x = self.norm(x)
        
        # chain of actions
        # self.q_proj(x) — Linear projection, (2, 5, 1024) → (2, 5, 1024)
        # .view(B, T, self.num_heads, self.head_dim) — ONLY Reshape, (2, 5, 1024) → (2, 5, 8, 128)
        # .transpose(1, 2) — (2, 5, 8, 128) → (2, 8, 5, 128), Now each head is a separate "batch" dimension. PyTorch can process all 8 heads in parallel with a single matrix multiply. The shape (2, 8, 5, 128) means: "2 sentences × 8 heads × 5 tokens × 128 features per token."
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5 # 128 ** -0.5 = 1/√128 ≈ 0.0884
        attn = (q @ k.transpose(-2, -1)) * scale # 5×5 attention score matrix 

        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool() # converts to True/False. The True positions are the future tokens — token 0 shouldn't see tokens 1-4, token 1 shouldn't see tokens 2-4, etc.
        attn = attn.masked_fill(mask, float('-inf')) # Everywhere the mask is True (future positions), fill with -infinity
        attn = F.softmax(attn, dim=-1) # converts scores to probabilities (sum to 1 per row). Since e^(-inf) = 0, masked positions get zero attention

        # chain of actions
        # attn @ v - (2, 8, 5, 5) @ (2, 8, 5, 128) → (2, 8, 5, 128), Weighted combination of value vectors.
        # .transpose(1, 2) — Swap heads and tokens back, (2, 8, 5, 128) → (2, 5, 8, 128)
        # .view(B, T, C) — Merge all heads back together, (2, 5, 8, 128) → (2, 5, 1024)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H)
        return self.out_proj(out) + residual # output = new changes + original


class MiniFfnBlock(nn.Module):
    """
    Mini feed-forward network block. The "Thinking" Block. If attention is about gathering information from other tokens, the FFN is about processing information within each token independently.

    The layer compute is simply: LayerNorm -> Linear -> Activation -> Linear -> Residual Add.
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = F.silu(x) #SiLU activation
        x = self.down_proj(x)
        x = x + residual #Residual connection
        return x


class MiniTransformerBlock(nn.Module):
    """
    A single transformer block combining self-attention and a feed-forward network.

    Sequentially applies MiniAttentionBlock then MiniFfnBlock, each with their own
    pre-norm and residual connection.
    """

    def __init__(self, hidden_size: int = 1024, num_heads: int = 8):
        super().__init__()
        self.attn = MiniAttentionBlock(hidden_size, num_heads)
        self.ffn = MiniFfnBlock(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.ffn(x)
        return x


class MiniTransformer(nn.Module):
    """
    A full (mini) decoder-only transformer language model.

    Combines token + positional embeddings, N stacked transformer blocks, a final
    LayerNorm, and a linear LM head that projects to vocab logits.
    Input: (B, T) token IDs. Output: (B, T, vocab_size) logits.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 1024, num_heads: int = 8,
                 num_layers: int = 6, max_seq_len: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.blocks = nn.ModuleList([
            MiniTransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)  # (B, T, hidden_size) → (B, T, vocab_size), logits raw scores, NOT probabilities
