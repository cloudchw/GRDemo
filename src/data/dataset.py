"""PyTorch Dataset and DataLoader for RankMixer CTR samples."""

import torch
from torch.utils.data import Dataset, DataLoader


class CTRDataset(Dataset):
    """Dataset for CTR samples produced by feature_engineering."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "user_id": torch.tensor(s["user_id"], dtype=torch.long),
            "item_id": torch.tensor(s["item_id"], dtype=torch.long),
            "category": torch.tensor(s["category"], dtype=torch.long),
            "user_interaction_count": torch.tensor(s["user_interaction_count"], dtype=torch.float),
            "user_avg_rating": torch.tensor(s["user_avg_rating"], dtype=torch.float),
            "price": torch.tensor(s["price"], dtype=torch.float),
            "item_seq": torch.tensor(s["item_seq"], dtype=torch.long),
            "seq_mask": torch.tensor(s["seq_mask"], dtype=torch.bool),
            "cross_stats": torch.tensor(s["cross_stats"], dtype=torch.float),
            "label": torch.tensor(s["label"], dtype=torch.float),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Custom collate to stack tensors."""
    keys = batch[0].keys()
    return {k: torch.stack([s[k] for s in batch]) for k in keys}


def make_dataloader(samples: list[dict], batch_size: int = 256, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from CTR samples."""
    dataset = CTRDataset(samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # CPU training
        pin_memory=False,
    )
