"""Per-token FFN for RankMixer (Paper Eq. 6-9).

Each token has its own independent FFN, preventing high-frequency features
from dominating low-frequency/long-tail signals.
"""

import torch
import torch.nn as nn


class PerTokenFFN(nn.Module):
    """Independent FFN for each token position."""

    def __init__(self, num_tokens: int, hidden_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * expansion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * expansion, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_tokens)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Kaiming initialization for FFN layers
        for ffn in self.ffns:
            nn.init.kaiming_normal_(ffn[0].weight, nonlinearity="linear")
            nn.init.kaiming_normal_(ffn[3].weight, nonlinearity="linear")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply per-token FFN.

        Args:
            tokens: (B, T, D)

        Returns:
            (B, T, D) with residual + LayerNorm
        """
        out = torch.stack(
            [self.ffns[i](tokens[:, i, :]) for i in range(len(self.ffns))],
            dim=1,
        )
        return self.norm(tokens + out)
