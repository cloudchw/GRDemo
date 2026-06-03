"""Output head: Mean pooling + prediction layer."""

import torch
import torch.nn as nn


class OutputHead(nn.Module):
    """Prediction head: Linear -> ReLU -> Dropout -> Linear -> Sigmoid."""

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict CTR probability.

        Args:
            x: (B, input_dim) pooled token representations

        Returns:
            (B,) sigmoid probabilities
        """
        return self.layers(x).squeeze(-1)
