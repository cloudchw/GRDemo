"""RankMixer Block and full model."""

import torch
import torch.nn as nn

from .token_mixing import MultiHeadTokenMixing
from .per_token_ffn import PerTokenFFN
from .output import OutputHead


class RankMixerBlock(nn.Module):
    """Single RankMixer block: Token Mixing + Per-token FFN."""

    def __init__(self, num_tokens: int, hidden_dim: int, num_heads: int, ffn_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.token_mixing = MultiHeadTokenMixing(num_tokens, hidden_dim, num_heads)
        self.ffn = PerTokenFFN(num_tokens, hidden_dim, ffn_expansion, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_mixing(tokens)
        x = self.ffn(x)
        return x


class RankMixer(nn.Module):
    """Full Dense RankMixer model for CTR prediction.

    Pipeline: Tokenization -> L x RankMixerBlock -> Mean Pooling -> Output
    """

    def __init__(
        self,
        num_tokens: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 3,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        # Embedding vocab sizes
        num_users: int = 16000,
        num_items: int = 10000,
        num_categories: int = 50,
        # Embedding dims
        user_emb_dim: int = 32,
        item_emb_dim: int = 32,
        category_emb_dim: int = 16,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.item_emb = nn.Embedding(num_items, item_emb_dim)
        self.category_emb = nn.Embedding(num_categories, category_emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.category_emb.weight)

        # Sequence encoder (attention-weighted pooling)
        self.seq_attention = nn.Linear(item_emb_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.seq_attention.weight)

        # Tokenization projection layers (one per token group)
        # Token 1: user profile (user_emb_dim + 2 numeric) -> D
        self.proj_token1 = nn.Linear(user_emb_dim + 2, hidden_dim)
        # Token 2: candidate item (item_emb_dim + 1 + category_emb_dim) -> D
        self.proj_token2 = nn.Linear(item_emb_dim + 1 + category_emb_dim, hidden_dim)
        # Token 3: sequence (item_emb_dim) -> D
        self.proj_token3 = nn.Linear(item_emb_dim, hidden_dim)
        # Token 4: cross features (2) -> D
        self.proj_token4 = nn.Linear(2, hidden_dim)

        # RankMixer blocks
        self.blocks = nn.ModuleList([
            RankMixerBlock(num_tokens, hidden_dim, num_heads, ffn_expansion, dropout)
            for _ in range(num_blocks)
        ])

        # Output head
        self.output_head = OutputHead(hidden_dim * num_tokens, dropout)

    def encode_sequence(self, item_seq: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        """Attention-weighted pooling over sequence of item embeddings."""
        embs = self.item_emb(item_seq)  # (B, seq_len, item_emb_dim)
        scores = self.seq_attention(embs).squeeze(-1)  # (B, seq_len)
        scores = scores.masked_fill(~seq_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~seq_mask, 0.0)
        return (embs * weights.unsqueeze(-1)).sum(dim=1)  # (B, item_emb_dim)

    def tokenize(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert raw features into T tokens of dimension D.

        Args:
            batch: dict with feature tensors

        Returns:
            (B, T, D) token representations
        """
        # Token 1: User Profile
        t1 = torch.cat([
            self.user_emb(batch["user_id"]),
            batch["user_interaction_count"].unsqueeze(-1),
            batch["user_avg_rating"].unsqueeze(-1),
        ], dim=-1)
        t1 = self.proj_token1(t1)

        # Token 2: Candidate Item
        t2 = torch.cat([
            self.item_emb(batch["item_id"]),
            batch["price"].unsqueeze(-1),
            self.category_emb(batch["category"]),
        ], dim=-1)
        t2 = self.proj_token2(t2)

        # Token 3: Sequence
        seq_repr = self.encode_sequence(batch["item_seq"], batch["seq_mask"])
        t3 = self.proj_token3(seq_repr)

        # Token 4: Cross Features
        t4 = self.proj_token4(batch["cross_stats"])

        # Stack: (B, T, D)
        return torch.stack([t1, t2, t3, t4], dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Full forward pass.

        Returns:
            (B,) CTR predictions (sigmoid probabilities)
        """
        tokens = self.tokenize(batch)  # (B, T, D)
        for block in self.blocks:
            tokens = block(tokens)

        # Mean pooling over tokens: (B, T, D) -> (B, T*D)
        pooled = tokens.reshape(tokens.size(0), -1)
        return self.output_head(pooled)
