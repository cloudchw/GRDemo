"""Feature engineering for RankMixer on Amazon Reviews 2023.

Handles: data loading, CTR sample construction, last-out split,
feature encoding, sequence building.
"""

import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd


def load_reviews(path: str) -> list[dict]:
    """Load review records from JSONL file.

    Returns list of {user_id, parent_asin, rating, timestamp}.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append({
                "user_id": r["user_id"],
                "parent_asin": r["parent_asin"],
                "rating": float(r["rating"]),
                "timestamp": int(r["timestamp"]),
            })
    return records


def load_metadata(path: str) -> dict[str, dict]:
    """Load item metadata from parquet file. Returns dict keyed by parent_asin."""
    df = pd.read_parquet(path, engine="fastparquet")

    # Parse price: strings like '6.99' or 'None'
    prices = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    # Use store as category (more discriminative than main_category)
    stores = df["store"].fillna("unknown")

    meta = {}
    for i, row in df.iterrows():
        parent_asin = row["parent_asin"]
        meta[parent_asin] = {
            "price": float(prices.iloc[i]),
            "store": str(stores.iloc[i]),
        }
    return meta


def kcore_filter(records: list[dict], k: int = 5) -> list[dict]:
    """Apply k-core filtering: keep users and items with >= k interactions.

    Iterates until stable.
    """
    while True:
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        for r in records:
            user_counts[r["user_id"]] += 1
            item_counts[r["parent_asin"]] += 1

        valid_users = {u for u, c in user_counts.items() if c >= k}
        valid_items = {it for it, c in item_counts.items() if c >= k}

        filtered = [
            r for r in records
            if r["user_id"] in valid_users and r["parent_asin"] in valid_items
        ]

        if len(filtered) == len(records):
            break
        records = filtered
    return records


def build_user_item_index(records: list[dict]) -> tuple[dict, dict, set, set]:
    """Build user -> sorted interactions, item -> users, and sets.

    Returns:
        user_seqs: {user_id: [(parent_asin, rating, timestamp), ...]} sorted by time
        item_users: {parent_asin: set(user_ids)}
        all_users: set of user_ids
        all_items: set of parent_asins
    """
    user_seqs = defaultdict(list)
    item_users = defaultdict(set)

    for r in records:
        user_seqs[r["user_id"]].append((r["parent_asin"], r["rating"], r["timestamp"]))
        item_users[r["parent_asin"]].add(r["user_id"])

    # Sort each user's interactions by timestamp
    for uid in user_seqs:
        user_seqs[uid].sort(key=lambda x: x[2])

    all_users = set(user_seqs.keys())
    all_items = set(item_users.keys())

    return dict(user_seqs), dict(item_users), all_users, all_items


class FeatureEncoder:
    """Builds vocabularies and normalizes features."""

    def __init__(self):
        self.user2idx = {}
        self.item2idx = {}
        self.cat2idx = {}
        self.price_mean = 0.0
        self.price_std = 1.0
        self.interaction_count_mean = 0.0
        self.interaction_count_std = 1.0

    def fit(
        self,
        user_seqs: dict,
        item_users: dict,
        metadata: dict[str, dict],
    ) -> "FeatureEncoder":
        """Build vocabularies and compute normalization stats."""
        # User vocabulary
        self.user2idx = {u: i for i, u in enumerate(sorted(user_seqs.keys()))}

        # Item vocabulary (only items in the filtered dataset)
        self.item2idx = {it: i for i, it in enumerate(sorted(item_users.keys()))}

        # Store vocabulary from metadata for active items only
        active_items = set(item_users.keys())
        store_counts = defaultdict(int)
        for item in active_items:
            if item in metadata:
                store_counts[metadata[item]["store"]] += 1

        # Merge rare stores (fewer than 3 items) into "other"
        cats = sorted(
            s for s, c in store_counts.items()
            if c >= 3 and s != "unknown"
        ) + ["other"]
        self.cat2idx = {c: i for i, c in enumerate(cats)}

        # Price normalization (only for active items)
        prices = [
            metadata[item]["price"]
            for item in active_items
            if item in metadata and metadata[item]["price"] > 0
        ]
        if prices:
            log_prices = np.log1p(prices)
            self.price_mean = float(np.mean(log_prices))
            self.price_std = float(np.std(log_prices)) + 1e-8

        # Interaction count normalization
        counts = [len(v) for v in user_seqs.values()]
        log_counts = np.log1p(counts)
        self.interaction_count_mean = float(np.mean(log_counts))
        self.interaction_count_std = float(np.std(log_counts)) + 1e-8

        return self

    @property
    def num_users(self) -> int:
        return len(self.user2idx)

    @property
    def num_items(self) -> int:
        return len(self.item2idx)

    @property
    def num_categories(self) -> int:
        return len(self.cat2idx)

    def encode_price(self, price: float) -> float:
        return (np.log1p(price) - self.price_mean) / self.price_std

    def encode_interaction_count(self, count: int) -> float:
        return (np.log1p(count) - self.interaction_count_mean) / self.interaction_count_std


def build_ctr_samples(
    user_seqs: dict,
    item_users: dict,
    metadata: dict[str, dict],
    encoder: FeatureEncoder,
    neg_ratio: int = 4,
    min_rating_pos: int = 4,
    seq_len: int = 20,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build CTR samples with last-out split.

    For each user:
    - Split interactions: newest positive -> test, 2nd newest -> valid, rest -> train
    - Positive: rating >= min_rating_pos
    - Negative: randomly sample un-interacted items (neg_ratio per positive)

    Returns:
        (train_samples, val_samples, test_samples) each a list of sample dicts
    """
    rng = random.Random(seed)
    all_items = list(item_users.keys())

    train_samples = []
    val_samples = []
    test_samples = []

    for uid, interactions in user_seqs.items():
        if uid not in encoder.user2idx:
            continue

        # Separate positives and negatives by rating
        positives = [(asin, r, ts) for asin, r, ts in interactions if r >= min_rating_pos]
        all_interacted = set(asin for asin, _, _ in interactions)

        if len(positives) < 2:
            continue  # Need at least 2 positives for val/test

        # Last-out split on positives
        test_pos = positives[-1:]
        val_pos = positives[-2:-1] if len(positives) >= 2 else []
        train_pos = positives[:-2] if len(positives) >= 2 else positives[:-1]

        # Negative item pool for this user
        neg_pool = [it for it in all_items if it not in all_interacted]

        def make_sample(uid: str, asin: str, label: int, hist_until_ts: int) -> "dict | None":
            if asin not in encoder.item2idx:
                return None

            # User features
            user_idx = encoder.user2idx[uid]
            all_user_interactions = user_seqs[uid]
            interaction_count = len(all_user_interactions)
            avg_rating = np.mean([r for _, r, _ in all_user_interactions])

            # Item features
            meta = metadata.get(asin, {})
            item_idx = encoder.item2idx[asin]
            cat = meta.get("store", "other")
            cat_idx = encoder.cat2idx.get(cat, encoder.cat2idx.get("other", 0))
            price = meta.get("price", 0.0)

            # Sequence: items before this interaction
            hist_items = [
                a for a, r, ts in all_user_interactions
                if ts < hist_until_ts
            ]
            recent = hist_items[-seq_len:]  # most recent N
            # Pad to seq_len
            seq_padded = recent + [""] * (seq_len - len(recent))
            seq_mask = [True] * len(recent) + [False] * (seq_len - len(recent))
            seq_indices = [encoder.item2idx.get(a, 0) for a in seq_padded]

            # Cross features: user-item co-occurrence count, time gap
            co_count = len([a for a in hist_items if a == asin])
            if hist_items:
                last_ts = max(ts for _, _, ts in all_user_interactions if ts < hist_until_ts)
                time_gap = np.log1p(max(hist_until_ts - last_ts, 0) / 1000.0)
            else:
                time_gap = 0.0

            return {
                "user_id": user_idx,
                "item_id": item_idx,
                "category": cat_idx,
                "user_interaction_count": encoder.encode_interaction_count(interaction_count),
                "user_avg_rating": float(avg_rating),
                "price": encoder.encode_price(price),
                "time_delta": float(time_gap),
                "item_seq": seq_indices,
                "seq_mask": seq_mask,
                "cross_stats": [float(co_count), float(time_gap)],
                "label": label,
            }

        # Helper to add positives + negatives for a split
        def add_split(pos_list: list, sample_list: list):
            for asin, r, ts in pos_list:
                s = make_sample(uid, asin, 1, ts)
                if s is not None:
                    sample_list.append(s)
                # Negative samples
                if neg_pool:
                    negs = rng.sample(neg_pool, min(neg_ratio, len(neg_pool)))
                    for neg_asin in negs:
                        ns = make_sample(uid, neg_asin, 0, ts)
                        if ns is not None:
                            sample_list.append(ns)

        add_split(train_pos, train_samples)
        add_split(val_pos, val_samples)
        add_split(test_pos, test_samples)

    rng.shuffle(train_samples)
    return train_samples, val_samples, test_samples


def prepare_data(
    data_dir: str = "./AmazonReviews2023",
    neg_ratio: int = 4,
    min_rating_pos: int = 4,
    seq_len: int = 20,
    seed: int = 42,
) -> tuple[list, list, list, FeatureEncoder]:
    """Full data preparation pipeline.

    Returns:
        (train, val, test, encoder)
    """
    from .download import download_all

    paths = download_all(data_dir)

    print("Loading reviews...")
    records = load_reviews(paths["reviews"])
    print(f"  Total reviews: {len(records)}")

    print("Loading metadata...")
    metadata = load_metadata(paths["metadata"])
    print(f"  Items with metadata: {len(metadata)}")

    print("Applying 3-core filtering...")
    records = kcore_filter(records, k=3)
    print(f"  After 3-core: {len(records)} reviews")

    print("Building user-item index...")
    user_seqs, item_users, all_users, all_items = build_user_item_index(records)
    print(f"  Users: {len(all_users)}, Items: {len(all_items)}")

    print("Fitting feature encoder...")
    encoder = FeatureEncoder()
    encoder.fit(user_seqs, item_users, metadata)
    print(f"  Vocab sizes: users={encoder.num_users}, items={encoder.num_items}, categories={encoder.num_categories}")

    print("Building CTR samples...")
    train, val, test = build_ctr_samples(
        user_seqs, item_users, metadata, encoder,
        neg_ratio=neg_ratio, min_rating_pos=min_rating_pos,
        seq_len=seq_len, seed=seed,
    )
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    return train, val, test, encoder
