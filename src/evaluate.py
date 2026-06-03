"""Evaluation script for RankMixer."""

import argparse

import torch

from model.rank_mixer import RankMixer
from data.feature_engineering import prepare_data
from data.dataset import make_dataloader
from train import evaluate, compute_auc


def ndcg_at_k(labels: torch.Tensor, preds: torch.Tensor, k: int = 10) -> float:
    """Compute NDCG@K for a single user's ranked list."""
    # Sort by predicted score descending
    sorted_indices = torch.argsort(preds, descending=True)[:k]
    sorted_labels = labels[sorted_indices]

    # DCG
    gains = (2 ** sorted_labels.float() - 1)
    discounts = torch.log2(torch.arange(2, len(sorted_labels) + 2, dtype=torch.float))
    dcg = (gains / discounts).sum().item()

    # Ideal DCG
    ideal_labels = torch.sort(labels, descending=True).values[:k]
    ideal_gains = (2 ** ideal_labels.float() - 1)
    ideal_discounts = torch.log2(torch.arange(2, len(ideal_labels) + 2, dtype=torch.float))
    idcg = (ideal_gains / ideal_discounts).sum().item()

    return dcg / idcg if idcg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate RankMixer")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/rankmixer_best.pt")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    print("Preparing data...")
    _, _, test_samples, encoder = prepare_data(
        data_dir=data_cfg["data_dir"],
        neg_ratio=data_cfg["neg_ratio"],
        min_rating_pos=data_cfg["min_rating_pos"],
        seq_len=data_cfg["seq_len"],
        seed=train_cfg["seed"],
    )

    test_loader = make_dataloader(test_samples, batch_size=train_cfg["batch_size"], shuffle=False)

    # Load model
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

    # Evaluate
    criterion = torch.nn.BCEWithLogitsLoss()
    metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test AUC:    {metrics['auc']:.4f}")
    print(f"Test LogLoss: {metrics['logloss']:.4f}")


if __name__ == "__main__":
    main()
