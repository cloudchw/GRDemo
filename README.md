# GRDemo — RankMixer 复现工程

基于论文 *"RankMixer: Scaling Up Ranking Models in Industrial Recommenders"*，在 **Amazon Reviews 2023 (All_Beauty)** 数据集上从零实现的 CTR 排序模型，面向 CPU 训练。

RankMixer 的核心思想是把推荐系统的异构特征**语义化分词（semantic tokenization）**为若干 token 组，用 **Token Mixing + Per-token FFN** 替代 self-attention 完成跨特征交互，兼顾表达力与线性计算开销。

## 模型架构

```
原始特征 ──► 4 个语义 Token（线性投影到 D=64）
         ──► [RankMixer Block × 3]   每个 Block = Token Mixing + Per-token FFN
         ──► Flatten(T·D) ──► 输出头（logits）
```

| Token | 特征组 | 内容 |
|-------|--------|------|
| 1 | 用户画像 | user_emb(32) + 交互次数 + 平均评分 |
| 2 | 候选商品 | item_emb(32) + price + category_emb(16) |
| 3 | 行为序列 | 注意力加权池化最近 20 个商品 |
| 4 | 交叉特征 | 共现次数 + 时间间隔 |

**参数量**：约 0.8M（默认配置）。

## 项目结构

```
GRDemo/
├── configs/
│   └── default.yaml            # 超参配置
├── src/
│   ├── data/
│   │   ├── download.py         # HuggingFace 镜像下载数据
│   │   ├── feature_engineering.py  # 样本构造 / 编码 / 序列构建
│   │   └── dataset.py          # CTRDataset / DataLoader
│   ├── model/
│   │   ├── rank_mixer.py       # 主模型 + tokenize()
│   │   ├── token_mixing.py     # 多头 Token Mixing
│   │   ├── per_token_ffn.py    # 每 token 独立 FFN
│   │   └── output.py           # 输出头
│   ├── train.py                # 训练入口（AUC / LogLoss）
│   ├── evaluate.py             # 评估脚本（含 NDCG@K）
│   └── infer.py                # 推理脚本（Top-K 推荐）
└── docs/superpowers/specs/     # 设计文档
```

## 环境依赖

- Python ≥ 3.10
- PyTorch（CPU 版即可）
- pandas, numpy, pyarrow, fastparquet, pyyaml

```bash
pip install torch pandas numpy pyarrow fastparquet pyyaml
```

## 快速开始

所有命令在 `src/` 目录下执行：

```bash
cd src

# 1. 训练（首次运行会自动下载 All_Beauty 数据到 ../AmazonReviews2023/）
python train.py

# 2. 评估 test 集 AUC / LogLoss
python evaluate.py

# 3. 为指定用户生成 Top-K 推荐（user_id 为数据集中的原始字符串 ID）
python infer.py --user_id "<raw_user_id>" --top_k 10
```

可使用自定义配置：

```bash
python train.py --config configs/default.yaml --seed 42
```

## 数据说明

- **数据集**：Amazon Reviews 2023 — `All_Beauty` 类目
- **k-core 过滤**：保留交互数 ≥ 3 的用户与商品（3-core）
- **样本构造**：rating ≥ 4 视为正样本，每个正样本配 4 个负样本（从未交互商品中随机采样）
- **划分**：每个用户最后 1 个正样本 → test，倒数第 2 个 → valid，其余 → train（last-out split）

## 关键超参（`configs/default.yaml`）

| 超参 | 值 | 说明 |
|------|-----|------|
| `hidden_dim` | 64 | token 隐藏维度 D |
| `num_tokens` | 4 | 特征组数 T |
| `num_heads` | 4 | Token Mixing 头数 |
| `num_blocks` | 3 | RankMixer 层数 L |
| `lr` | 1e-3 | AdamW 学习率 |
| `weight_decay` | 1e-4 | L2 正则 |
| `warmup_steps` | 100 | 余弦退火热身步数 |
| `epochs` | 20 | 最大训练轮数（early stop patience=3） |
| `batch_size` | 256 | CPU 内存友好 |

## 评估指标

- **AUC**（基于 Wilcoxon-Mann-Whitney 统计量实现，无 sklearn 依赖）
- **LogLoss**
- **NDCG@K**（`evaluate.py` 提供）

预期参考：在 All_Beauty 上 test AUC 通常可达 **0.65 ~ 0.70** 量级（CPU / 默认配置，受负采样随机性影响）。

## 设计文档

详细设计（特征表、超参表、模块说明）见 [`docs/superpowers/specs/2026-04-24-rankmixer-reproduction-design.md`](docs/superpowers/specs/2026-04-24-rankmixer-reproduction-design.md)。
