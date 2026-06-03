"""RankMixer training script."""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model.rank_mixer import RankMixer
from data.feature_engineering import prepare_data
from data.dataset import make_dataloader


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    num_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")

        logits = model(batch)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        all_labels.append(labels.cpu())
        all_preds.append(torch.sigmoid(logits).cpu())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    labels = torch.cat(all_labels)
    preds = torch.cat(all_preds)

    # AUC
    auc = compute_auc(labels, preds)
    # LogLoss
    logloss = -(labels * torch.log(preds + 1e-12) + (1 - labels) * torch.log(1 - preds + 1e-12)).mean().item()

    return {"loss": avg_loss, "auc": auc, "logloss": logloss}


def compute_auc(labels: torch.Tensor, preds: torch.Tensor) -> float:
    """Compute AUC without sklearn dependency."""
    # Simple implementation using Wilcoxon-Mann-Whitney statistic
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_preds = preds[pos_mask]
    neg_preds = preds[neg_mask]

    if len(pos_preds) == 0 or len(neg_preds) == 0:
        return 0.5

    # Efficient vectorized computation
    # Count pairs where positive score > negative score + 0.5 * ties
    comparisons = (pos_preds.unsqueeze(1) > neg_preds.unsqueeze(0)).float().sum()
    ties = (pos_preds.unsqueeze(1) == neg_preds.unsqueeze(0)).float().sum()
    auc = (comparisons + 0.5 * ties) / (len(pos_preds) * len(neg_preds))
    return auc.item()


def main():
    parser = argparse.ArgumentParser(description="Train RankMixer")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    output_cfg = config["output"]

    # Seed
    torch.manual_seed(train_cfg["seed"])
    os.makedirs(output_cfg["dir"], exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Preparing data...")
    train_samples, val_samples, test_samples, encoder = prepare_data(
        data_dir=data_cfg["data_dir"],
        neg_ratio=data_cfg["neg_ratio"],
        min_rating_pos=data_cfg["min_rating_pos"],
        seq_len=data_cfg["seq_len"],
        seed=train_cfg["seed"],
    )

    train_loader = make_dataloader(train_samples, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = make_dataloader(val_samples, batch_size=train_cfg["batch_size"], shuffle=False)
    test_loader = make_dataloader(test_samples, batch_size=train_cfg["batch_size"], shuffle=False)

    # Model
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    total_steps = len(train_loader) * train_cfg["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, train_cfg["warmup_steps"], total_steps
    )

    # Training loop
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            train_cfg["max_grad_norm"],
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1}/{train_cfg['epochs']} "
            f"({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val LogLoss: {val_metrics['logloss']:.4f}"
        )

        # Early stopping on AUC
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save(model.state_dict(), output_cfg["checkpoint"])
            print(f"  -> Saved best model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Final test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(output_cfg["checkpoint"], weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test LogLoss: {test_metrics['logloss']:.4f}")


if __name__ == "__main__":
    main()
