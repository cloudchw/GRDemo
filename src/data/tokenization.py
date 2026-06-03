"""Tokenization: feature grouping -> unified tokens via linear projection.

This is handled inside the RankMixer model's tokenize() method.
This module provides helper utilities for token dimension calculation.
"""


def get_token_dims(
    user_emb_dim: int = 32,
    item_emb_dim: int = 32,
    category_emb_dim: int = 16,
) -> dict[str, int]:
    """Calculate raw feature dimensions per token group.

    Returns dict mapping token name -> concatenated feature dimension
    (before projection to hidden_dim).
    """
    return {
        "token1": user_emb_dim + 2,         # user_emb + interaction_count + avg_rating
        "token2": item_emb_dim + 1 + category_emb_dim,  # item_emb + price + category_emb
        "token3": item_emb_dim,              # sequence (weighted pool of item embs)
        "token4": 2,                         # cross_stats (co_count, time_gap)
    }
