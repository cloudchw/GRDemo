"""Batch inference script for RankMixer."""

import argparse

import torch

from model.rank_mixer import RankMixer


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


def main():
    parser = argparse.ArgumentParser(description="RankMixer Inference")
    parser.add_argument("--checkpoint", type=str, default="outputs/rankmixer_best.pt")
    parser.add_argument("--user_id", type=int, required=True, help="User index")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {args.checkpoint}...")

    # NOTE: This is a demo script. In production, you'd load the encoder
    # and build the full feature dict. Here we show the inference API.
    print("Inference demo - requires trained model and encoder.")
    print(f"Would recommend top-{args.top_k} items for user {args.user_id}")


if __name__ == "__main__":
    main()
