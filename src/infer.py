"""Batch inference script for RankMixer."""

import argparse

import torch

from model.rank_mixer import RankMixer
from data.feature_engineering import prepare_data


def infer_user(
    model: RankMixer,
    user_features: dict,
    candidate_items: list[dict],
    device: torch.device,
    batch_size: int = 100,
) -> list[tuple[int, float]]:
    """Score candidate items for a user, return sorted (item_idx, score) pairs."""
    model.eval()
    results = []

    with torch.no_grad():
        for i in range(0, len(candidate_items), batch_size):
            batch_items = candidate_items[i : i + batch_size]
            B = len(batch_items)

            # Expand user features to batch
            batch = {
                "user_id": user_features["user_id"].expand(B),
                "user_interaction_count": user_features["user_interaction_count"].expand(B),
                "user_avg_rating": user_features["user_avg_rating"].expand(B),
                "item_seq": user_features["item_seq"].expand(B, -1),
                "seq_mask": user_features["seq_mask"].expand(B, -1),
                "item_id": torch.tensor([it["item_id"] for it in batch_items], dtype=torch.long, device=device),
                "category": torch.tensor([it["category"] for it in batch_items], dtype=torch.long, device=device),
                "price": torch.tensor([it["price"] for it in batch_items], dtype=torch.float, device=device),
                "cross_stats": torch.stack([it["cross_stats"] for it in batch_items]).to(device),
            }

            logits = model(batch)
            scores = torch.sigmoid(logits).cpu().tolist()

            for j, item in enumerate(batch_items):
                results.append((item["item_id"], scores[j]))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def build_user_features(user_idx, interactions, user_seqs, metadata, encoder, seq_len, device):
    """Build the user-side feature tensors shared across all candidates.

    Mirrors the per-sample construction in feature_engineering.build_ctr_samples,
    using the user's full interaction history as the sequence.
    """
    interaction_count = len(interactions)
    avg_rating = float(sum(r for _, r, _ in interactions) / max(interaction_count, 1))

    # Sequence: most recent seq_len items (oldest -> newest, padded at the front)
    recent = [asin for asin, _, _ in interactions][-seq_len:]
    seq_indices = [encoder.item2idx.get(a, 0) for a in recent]
    seq_mask = [True] * len(seq_indices)
    # Left-pad so the most recent items align to the tail (consistent with training)
    pad = seq_len - len(seq_indices)
    if pad > 0:
        seq_indices = [0] * pad + seq_indices
        seq_mask = [False] * pad + seq_mask

    return {
        "user_id": torch.tensor(user_idx, dtype=torch.long, device=device),
        "user_interaction_count": torch.tensor(
            encoder.encode_interaction_count(interaction_count), dtype=torch.float, device=device
        ),
        "user_avg_rating": torch.tensor(avg_rating, dtype=torch.float, device=device),
        "item_seq": torch.tensor(seq_indices, dtype=torch.long, device=device).unsqueeze(0),
        "seq_mask": torch.tensor(seq_mask, dtype=torch.bool, device=device).unsqueeze(0),
    }


def build_candidate_items(user_interacted, item_users, metadata, encoder):
    """Build candidate list: all catalog items not already interacted with."""
    candidates = []
    for asin in item_users.keys():
        if asin in user_interacted:
            continue
        if asin not in encoder.item2idx:
            continue
        meta = metadata.get(asin, {})
        cat = meta.get("store", "other")
        cat_idx = encoder.cat2idx.get(cat, encoder.cat2idx.get("other", 0))
        price = encoder.encode_price(meta.get("price", 0.0))
        # cross_stats use no co-occurrence and zero time gap for unseen items
        cross_stats = torch.tensor([0.0, 0.0], dtype=torch.float)
        candidates.append({
            "item_id": encoder.item2idx[asin],
            "category": cat_idx,
            "price": price,
            "cross_stats": cross_stats,
            "_asin": asin,
        })
    return candidates


def main():
    parser = argparse.ArgumentParser(description="RankMixer Inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/rankmixer_best.pt")
    parser.add_argument("--user_id", type=str, required=True, help="Raw user_id (string) to recommend for")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Rebuild data pipeline to recover encoder + indexes (no model training needed)
    print("Preparing data (encoder + indexes)...")
    _, _, _, encoder = prepare_data(
        data_dir=data_cfg["data_dir"],
        neg_ratio=data_cfg["neg_ratio"],
        min_rating_pos=data_cfg["min_rating_pos"],
        seq_len=data_cfg["seq_len"],
        seed=train_cfg["seed"],
    )

    # Reload user_seqs / item_users / metadata via the same pipeline
    from data.feature_engineering import (
        download_all, load_reviews, load_metadata, kcore_filter, build_user_item_index,
    )
    paths = download_all(data_cfg["data_dir"])
    records = load_reviews(paths["reviews"])
    records = kcore_filter(records, k=3)
    user_seqs, item_users, _, _ = build_user_item_index(records)
    metadata = load_metadata(paths["metadata"])

    if args.user_id not in encoder.user2idx:
        print(f"ERROR: user_id '{args.user_id}' not found in vocabulary.")
        return
    user_idx = encoder.user2idx[args.user_id]
    interactions = user_seqs.get(args.user_id, [])
    if not interactions:
        print(f"ERROR: no interaction history for user '{args.user_id}'.")
        return

    print(f"Loading model from {args.checkpoint}...")
    model = RankMixer(
        num_tokens=model_cfg["num_tokens"],
        hidden_dim=model_cfg["hidden_dim"],
        num_heads=model_cfg["num_heads"],
        num_blocks=model_cfg["num_blocks"],
        ffn_expansion=model_cfg["ffn_expansion"],
        dropout=model_cfg["dropout"],
        num_users=encoder.num_users,
        num_items=encoder.num_items,
        num_categories=encoder.num_categories,
        user_emb_dim=model_cfg["user_emb_dim"],
        item_emb_dim=model_cfg["item_emb_dim"],
        category_emb_dim=model_cfg["category_emb_dim"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    # Build user + candidate features
    user_features = build_user_features(
        user_idx, interactions, user_seqs, metadata, encoder, data_cfg["seq_len"], device,
    )
    user_interacted = {asin for asin, _, _ in interactions}
    candidates = build_candidate_items(user_interacted, item_users, metadata, encoder)
    print(f"Scoring {len(candidates)} candidate items for user '{args.user_id}'...")

    results = infer_user(model, user_features, candidates, device, batch_size=100)

    print(f"\nTop-{args.top_k} recommendations for user '{args.user_id}':")
    for rank, (item_idx, score) in enumerate(results[: args.top_k], 1):
        # Reverse-lookup the asin for display
        asin = next((c["_asin"] for c in candidates if c["item_id"] == item_idx), "?")
        print(f"  {rank:2d}. {asin}  (item_idx={item_idx}, score={score:.4f})")


if __name__ == "__main__":
    main()
