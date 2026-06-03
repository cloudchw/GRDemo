# RankMixer Reproduction Design

Reproduce the Dense RankMixer ranking model from "RankMixer: Scaling Up Ranking Models in Industrial Recommenders" on the Amazon Reviews 2023 All_Beauty dataset, with CPU training.

## 1. Overview

| Dimension | Decision |
|-----------|----------|
| Dataset | Amazon Reviews 2023 - All_Beauty (5-core: ~16K users, ~10K items, ~70K interactions) |
| Hardware | Windows CPU training |
| Framework | PyTorch |
| Task | CTR ranking (binary classification: click / no-click) |
| Model | Dense RankMixer (no MoE) |
| Approach | From-scratch implementation with custom feature pipeline |

## 2. Dataset & Feature Engineering

### 2.1 Data Source

Amazon Reviews 2023 All_Beauty category:
- Rating records: `user_id, parent_asin, rating, timestamp`
- Item metadata: `title, price, store, categories, features, description`
- User behavior sequences: chronologically ordered interaction history

### 2.2 CTR Sample Construction

- **Positive samples**: rating >= 4 treated as "click"
- **Negative samples**: for each user, randomly sample un-interacted items (neg:pos ratio = 4:1)
- **Data split**: last-out split (newest positive sample for test, second newest for validation, rest for training)

### 2.3 Feature Grouping & Tokenization

Align with the paper's semantic-based tokenization approach. Features are grouped into 4 semantically coherent clusters, each projected into a unified token:

| Token Group | Features | Semantics |
|-------------|----------|-----------|
| Token 1 - User Profile | user_id embedding, historical interaction count (numeric), user average rating (numeric) | User-side information |
| Token 2 - Candidate Item | item_id embedding, price (numeric), category (categorical) | Item-side information |
| Token 3 - Sequence | Weighted pooling of user's recent N interacted items | Temporal user interest |
| Token 4 - Cross Features | User-item interaction statistics, time delta | Cross-space interaction |

Each token is linearly projected to a unified dimension D via a learned linear layer `Proj(e_input[d*(i-1):d*i])` (Paper Eq. 2), yielding T=4 tokens total.

### 2.4 Embedding Dimensions

| Feature | Type | Embedding Dim |
|---------|------|--------------|
| user_id | Categorical (vocabulary ~16K) | 32 |
| item_id | Categorical (vocabulary ~10K) | 32 |
| category | Categorical (vocabulary ~50) | 16 |
| user_interaction_count | Numeric (log-normalized) | 1 |
| user_avg_rating | Numeric (standardized) | 1 |
| price | Numeric (log-normalized) | 1 |
| time_delta | Numeric (log-normalized) | 1 |
| sequence_repr | Dense (weighted pool of item embs) | 32 |
| cross_stats | Numeric (co-occurrence count, time gap) | 2 |

Each token group concatenates its features, then projects to D=64 via a linear layer.

### 2.5 Sequence Feature Processing

User's recent N=20 interactions are processed as:
1. Look up item_id embeddings for each historical item
2. Apply attention-weighted pooling: weight = softmax(W_att * item_emb), where W_att is learnable
3. Result: a 32-dim dense vector representing temporal user interest

## 3. Model Architecture

### 3.1 Overall Structure

```
Input Embeddings (4 feature groups concatenated)
    |
Tokenization (linear projection -> 4 tokens of D dims each)
    |
[ RankMixer Block x L layers ]
    |
    +-- Multi-head Token Mixing:
    |     1. Split each token into H=4 heads
    |     2. Reorder heads across tokens (parameter-free channel shuffle)
    |     3. Linear projection to restore dimension
    |     4. Add & LayerNorm
    |
    +-- Per-token FFN:
          Each token processed by independent FFN:
          FFN_t(x) = W2(GELU(W1 * x))
          W1: D -> 4D, W2: 4D -> D
          Parameters NOT shared across tokens
          Add & LayerNorm
    |
Mean Pooling (average 4 tokens)
    |
Output Layer -> Sigmoid -> CTR prediction
```

### 3.2 Hyperparameters (CPU Small Model)

| Parameter | Value | Notes |
|-----------|-------|-------|
| T (num tokens) | 4 | 4 semantic feature groups |
| D (hidden dim) | 64 | Keep compute manageable on CPU |
| H (num heads) | 4 | Equals T for residual dimension consistency |
| L (num blocks) | 3 | Sufficient for small dataset |
| FFN expansion | 4x | 64 -> 256 -> 64 |
| Dropout | 0.1 | Regularization |
| Embedding dims | user_id=32, item_id=32, category=16 | Vocabulary-size dependent |
| Total params | ~0.8M | CPU-friendly (includes embeddings ~0.6M + model ~0.2M) |

### 3.3 Multi-head Token Mixing (Paper Eq. 3-5)

- Input: T=4 tokens, each D=64-dim
- **SplitHead**: each token x_t is split into H=4 heads: x_t = [x_t^(1) || x_t^(2) || ... || x_t^(H)], each head is D/H=16 dim
- **Concat across tokens**: for head h, concatenate the h-th head from ALL tokens: s_h = Concat(x_1^(h), x_2^(h), ..., x_T^(h)) -> shape (T * D/H) = (4 * 16) = 64 dim
- Output S is stacked by all H=4 shuffled tokens s_1, ..., s_H -> shape (H, T*D/H) = (4, 64)
- **Key insight**: this is NOT a channel shuffle. It collects sub-vectors from different tokens to form new mixed representations, enabling cross-token feature interaction without self-attention's quadratic cost and without any learnable parameters
- H=T=4 ensures output has same number of tokens as input, enabling residual connection

### 3.4 Per-token FFN (Paper Eq. 6-9)

- Standard Transformer: all tokens share one FFN
- RankMixer: token 1 (user) uses FFN_1, token 2 (item) uses FFN_2, etc. - each independent
- Prevents high-frequency features from dominating low-frequency/long-tail signals (Paper Section 3.3.2)
- Increases model capacity while keeping FLOPs unchanged vs shared FFN

## 4. Training Plan

| Item | Configuration |
|------|--------------|
| Loss function | BCE (Binary Cross Entropy) |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| LR schedule | CosineAnnealing, warmup 500 steps |
| Batch size | 256 (CPU memory-friendly) |
| Epochs | 20 (early stopping patience=3 on validation AUC) |
| Negative sampling | 4 negatives per positive, re-sample each epoch |
| Gradient clipping | max_norm=5.0 |
| Embedding init | Xavier uniform for embeddings, Kaiming for FFN |
| Random seed | 42 (reproducibility) |

## 5. Evaluation Metrics

- **AUC** (Area Under ROC Curve) - primary metric
- **LogLoss** - calibration quality
- **NDCG@10** - ranking quality

## 6. Inference Plan

- Model export: `torch.save` for state_dict
- Inference mode: `model.eval()` + `torch.no_grad()`
- Given a user, score all candidate items -> sort by score descending
- Batch inference: score 100 candidate items at a time to avoid per-item overhead

## 7. Project Structure

```
D:/GitHub/RankMixer/
+-- AmazonReviews2023/              # Existing data source
+-- src/
|   +-- data/
|   |   +-- download.py             # Download All_Beauty data
|   |   +-- feature_engineering.py  # Feature extraction & encoding
|   |   +-- tokenization.py         # Feature grouping -> Tokens
|   |   +-- dataset.py              # PyTorch Dataset/DataLoader
|   +-- model/
|   |   +-- embedding.py            # Feature embedding layers
|   |   +-- token_mixing.py         # Multi-head Token Mixing
|   |   +-- per_token_ffn.py        # Per-token FFN
|   |   +-- rank_mixer.py           # RankMixer Block + full model
|   |   +-- output.py               # Mean Pooling + output head
|   +-- train.py                    # Training entry point
|   +-- evaluate.py                 # Evaluation script
|   +-- infer.py                    # Inference script
+-- configs/
|   +-- default.yaml                # Hyperparameter config
+-- outputs/                        # Model checkpoints, logs
+-- docs/
    +-- superpowers/specs/          # Design docs
```
