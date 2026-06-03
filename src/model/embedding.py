"""Feature embedding layers for RankMixer."""

import torch
import torch.nn as nn


class FeatureEmbedding(nn.Module):
    """Embedding layer that encodes categorical, numeric, and sequence features.

    Outputs concatenated embeddings ready for tokenization (4 groups).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        user_emb_dim: int = 32,
        item_emb_dim: int = 32,
        category_emb_dim: int = 16,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.item_emb = nn.Embedding(num_items, item_emb_dim)
        self.category_emb = nn.Embedding(num_categories, category_emb_dim)

        # Initialize
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.category_emb.weight)

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        category: torch.Tensor,
        user_interaction_count: torch.Tensor,
        user_avg_rating: torch.Tensor,
        price: torch.Tensor,
        time_delta: torch.Tensor,
        sequence_repr: torch.Tensor,
        cross_stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return embeddings grouped by token.

        Returns dict with keys: token1, token2, token3, token4
        """
        token1 = torch.cat([
            self.user_emb(user_id),
            user_interaction_count.unsqueeze(-1),
            user_avg_rating.unsqueeze(-1),
        ], dim=-1)

        token2 = torch.cat([
            self.item_emb(item_id),
            price.unsqueeze(-1),
            self.category_emb(category),
        ], dim=-1)

        # token3: sequence representation (already dense from weighted pooling)
        token3 = sequence_repr

        # token4: cross features
        token4 = cross_stats

        return {
            "token1": token1,  # user profile: (B, user_emb_dim + 2)
            "token2": token2,  # candidate item: (B, item_emb_dim + 1 + category_emb_dim)
            "token3": token3,  # sequence: (B, item_emb_dim)
            "token4": token4,  # cross: (B, 2)
        }


class SequenceEncoder(nn.Module):
    """Attention-weighted pooling over user's recent item interactions."""

    def __init__(self, item_emb: nn.Embedding):
        super().__init__()
        self.item_emb = item_emb
        emb_dim = item_emb.embedding_dim
        self.attention_w = nn.Linear(emb_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attention_w.weight)

    def forward(self, item_seq: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        """Encode sequence via attention-weighted pooling.

        Args:
            item_seq: (B, seq_len) item indices, 0-padded
            seq_mask: (B, seq_len) bool, True for valid positions

        Returns:
            (B, emb_dim) weighted pooled representation
        """
        embs = self.item_emb(item_seq)  # (B, seq_len, emb_dim)
        scores = self.attention_w(embs).squeeze(-1)  # (B, seq_len)
        scores = scores.masked_fill(~seq_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, seq_len)
        weights = weights.masked_fill(~seq_mask, 0.0)
        pooled = (embs * weights.unsqueeze(-1)).sum(dim=1)  # (B, emb_dim)
        return pooled
