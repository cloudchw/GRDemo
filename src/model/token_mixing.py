"""Multi-head Token Mixing for RankMixer.

Implements the parameter-free cross-token mixing from the paper (Eq. 3-5).
"""

import torch
import torch.nn as nn


class MultiHeadTokenMixing(nn.Module):
    """Multi-head token mixing with channel shuffle (parameter-free mixing).

    Splits each token into H heads, reorders heads across tokens to enable
    cross-token feature interaction without self-attention's quadratic cost.
    """

    def __init__(self, num_tokens: int, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert num_heads == num_tokens, "For residual consistency, H must equal T"

        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projection after mixing (Eq. 5)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply multi-head token mixing.

        Args:
            tokens: (B, T, D) input token representations

        Returns:
            (B, T, D) mixed tokens with residual connection
        """
        B, T, D = tokens.shape
        H = self.num_heads
        hd = self.head_dim

        # Split each token into H heads: (B, T, D) -> (B, T, H, hd)
        split = tokens.view(B, T, H, hd)

        # Reorder: collect h-th head from all tokens -> s_h = (B, T*hd)
        # Stack all heads: (B, H, T*hd) = (B, H, D) since T*hd = D
        shuffled = split.permute(0, 2, 1, 3).contiguous()  # (B, H, T, hd)
        shuffled = shuffled.view(B, H, T * hd)  # (B, H, D)

        # Project back and reshape
        out = self.proj(shuffled)  # (B, H, D)
        out = out.view(B, H, T, hd)  # (B, H, T, hd)
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, T, H, hd)
        out = out.view(B, T, D)  # (B, T, D)

        # Residual + LayerNorm
        return self.norm(tokens + out)
