# GQA-Context-Bench

![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange)
![Python](https://img.shields.io/badge/Python-3-blue)
![Dataset](https://img.shields.io/badge/Dataset-TinyStories-lightgrey)

## Executive Summary

GQA-Context-Bench is a controlled, from-scratch ablation of four attention topologies: **MHA (16 KV heads)**, **GQA-4 (4 KV heads)**, **GQA-8 (2 KV heads)** and **MQA (1 KV head)**. All four are ~20M-parameter decoder-only Transformers trained on TinyStories at $L_{\text{train}}=1024$ and evaluated zero-shot at 1024, 2048, 4096 and 8192 tokens ($8\times$ extrapolation) with dynamic NTK-aware RoPE. The benchmark measures long-context perplexity, KV-cache size, peak GPU memory, and prefill/decode throughput.

At 8192 tokens, GQA-4 cuts the KV cache by 75.0% (96.75 MiB to 24.19 MiB) and peak VRAM by 25.4% (338.5 MiB to 252.5 MiB). Its perplexity is 33.60 against 30.01 for MHA. MQA reaches a 32.0% peak VRAM reduction at 34.27 PPL. Batch-1 decode speed shows no reliable ordering between variants (Section 5).

---

## 1. The Problem and Systems Bottleneck

**KV-cache memory wall.** During autoregressive inference every layer stores K and V for every past token. Per sequence this grows as $O(L)$:

$$\text{KV bytes} = 2 \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot L \cdot d_{\text{head}} \cdot \text{bytes}_{\text{dtype}}$$

For the MHA model here (6 layers, 16 KV heads, head_dim 32, fp16) this is 12 KiB per token. At 8192 context the cache reaches 96.75 MiB, which is larger than the model weights themselves. Reducing `n_kv_heads` shrinks this term proportionally, and the query side is never cached.

**RoPE extrapolation breakdown.** A model trained at $L_{\text{train}}=1024$ has never seen rotary phases beyond position 1024. Without frequency rescaling, perplexity degrades quickly past that length. Even with NTK-aware scaling, every variant here loses substantial quality at 8192 (Section 5).

**Research question.** *How much KV-cache and peak VRAM can be saved via GQA/MQA before long-context perplexity degrades, and what actually bottlenecks latency at small parameter scales?*

---

## 2. Architecture and Methods

### Model (`model.py`)

From-scratch PyTorch decoder-only Transformer, no external model libraries.

| Hyperparameter | Value |
|---|---|
| `n_layers` | 6 |
| `dim` (d_model) | 512 |
| `n_heads` (query heads) | 16 (head_dim = 32) |
| `n_kv_heads` | 16 (MHA), 4 (GQA-4), 2 (GQA-8), 1 (MQA) |
| `vocab_size` | 2048 |
| MLP | SwiGLU, hidden dim 1408 (2/3 x 4 x 512 = 1365, rounded up to a multiple of 64) |
| Normalization | RMSNorm (eps 1e-6), pre-norm |
| Positional encoding | RoPE, theta = 10000 |
| Embeddings | tied input/output, no biases, no dropout |
| `max_seq_len` ($L_{\text{train}}$) | 1024 |

Parameter counts, computed from this configuration:

| Variant | Parameters |
|---|---|
| MHA | 20,322,816 |
| GQA-4 | 17,963,520 |
| GQA-8 | 17,570,304 |
| MQA | 17,373,696 |

`n_heads=16` is used so that GQA-8, GQA-4 and MQA are three distinct topologies. With `n_heads=8`, "8 query heads per KV head" would collapse into MQA. Removing KV heads is not compensated by extra width, so the GQA/MQA variants have fewer parameters than MHA (see Limitations).

### Dynamic NTK-aware RoPE

For a sequence of length $L > L_{\text{train}}$, the rotary base is rescaled:

$$s = \frac{L}{L_{\text{train}}}, \qquad \theta' = \theta \cdot s^{\,d_{\text{head}}/(d_{\text{head}}-2)}$$

with $\theta = 10000$ and $d_{\text{head}} = 32$. This stretches low frequencies to cover the longer range while keeping high frequencies sharp, and it spans $2\times$ to $8\times$ extrapolation here. For $L \le 1024$ the scale is 1 and the encoding is identical to vanilla RoPE.

### Systems optimizations

- **Native grouped attention.** `F.scaled_dot_product_attention(..., enable_gqa=True)` consumes the compact K/V tensors directly, so K/V are never expanded back to 16 heads in memory (requires PyTorch 2.5+).
- **Mixed precision.** `fp16` training with `GradScaler`, and `fp16` inference.
- **Pre-allocated `fp16` KV cache.** One buffer of shape `(n_layers, batch, n_kv_heads, max_seq_len, head_dim)` each for K and V. Cache size is exact and independent of allocator behavior.

### Training recipe (`train.py`)

AdamW (betas 0.9/0.95, weight decay 0.1), peak LR 3e-4, 100 warmup steps, cosine decay, gradient clipping at 1.0, 3000 steps, batch size 32, block size 1024 (about 98.3M training tokens per variant), seed 42.

---

## 3. Dataset and Evaluation Protocol

**Data.** The first 20M characters of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (train split), one story per row. If the download fails, `data.py` falls back to a synthetic corpus, so check the training log for `[data] Loaded real TinyStories`.

**Tokenizer.** A byte-level BPE tokenizer with 2048 tokens (256 bytes plus 1792 merges), trained on the first 2M characters of the train split. Merge rules are stored in fixed-size PyTorch tensors. Encoding uses a linked-list plus min-heap implementation (`encode_bulk_fast`), which the notebook reports as about 2 minutes for a 20M-character corpus.

**Content-hashed whole-story split.** Each story is assigned to train or validation (5% validation) by hashing its text together with the seed. Two properties follow:

1. No story is cut across the boundary. A character-position split would put a story's opening in train and its continuation in validation, so validation windows would contain text the model partly memorized.
2. Assignment depends only on the story's content, not on its position in the loaded list. `train.py` and `evaluate.py` may load different amounts of data (`--max_chars`), and a position-based split would silently give them different held-out sets.

### Benchmark suites (`evaluate.py`)

1. **Long-context perplexity ($1024 \to 8192$).** For each context length, mean cross-entropy over 10 evenly spaced held-out windows, exponentiated.
2. **KV-cache and peak GPU VRAM scaling.** Exact KV-cache size, and `torch.cuda.max_memory_allocated` over a full prefill plus decode pass (weights, KV cache and activations).
3. **Prefill and decode throughput/latency.** Batch size 1. Prefill of `context_len` tokens into the pre-allocated cache, then 64 greedy decode steps. Median of 3 timed runs after one warm-up run. Speed tests use random token IDs.

The KV cache is allocated for `context_len + 64` positions, so "L=1024" rows reflect 1088 slots. Memory values in the CSV are MiB (divided by 1024²).

---

## 4. Quantitative Results

All numbers below come from `results/eval_results.csv` (rounded as in `summary_table.png`). Reductions are relative to MHA at the same context length.

| Model | KV heads | L | PPL | KV cache (MiB) | Peak VRAM (MiB) | Peak VRAM reduction vs MHA | KV reduction vs MHA |
|---|---|---|---|---|---|---|---|
| MHA | 16 | 1024 | 16.35 | 12.8 | 156.3 | 0.0% | 0.0% |
| GQA-4 | 4 | 1024 | 16.02 | 3.2 | 132.7 | 15.1% | 75.0% |
| GQA-8 | 2 | 1024 | 15.93 | 1.6 | 127.6 | 18.3% | 87.5% |
| MQA | 1 | 1024 | 16.03 | 0.8 | 125.7 | 19.6% | 93.8% |
| MHA | 16 | 8192 | 30.01 | 96.8 | 338.5 | 0.0% | 0.0% |
| GQA-4 | 4 | 8192 | 33.60 | 24.2 | 252.5 | 25.4% | 75.0% |
| GQA-8 | 2 | 8192 | 34.97 | 12.1 | 237.4 | 29.9% | 87.5% |
| MQA | 1 | 8192 | 34.27 | 6.0 | 230.1 | 32.0% | 93.8% |

### Perplexity

![Perplexity Curves](assets/perplexity_curves.png)

- Inside $L_{\text{train}}$ (1024) all variants are within 15.93 to 16.35 PPL. MHA is highest (16.35) and GQA-8 lowest (15.93).
- Every variant degrades sharply past 4096. At 8192, MHA reaches 30.01, GQA-4 33.60, MQA 34.27 and GQA-8 34.97. Relative to MHA that is +11.9%, +14.2% and +16.5%.
- From 1024 to 8192, PPL grows 1.84x for MHA, 2.10x for GQA-4, 2.14x for MQA and 2.20x for GQA-8. MHA extrapolates best.

### KV cache and peak VRAM

![VRAM Scaling](assets/vram_scaling.png)

- The KV cache at 8192 is 96.75 MiB (MHA), 24.19 MiB (GQA-4), 12.09 MiB (GQA-8) and 6.05 MiB (MQA). The reductions are exactly 75.0%, 87.5% and 93.8%, as expected from `n_kv_heads / 16`.
- Peak VRAM reductions grow with context. GQA-4: 15.1% (1024), 18.2% (2048), 22.0% (4096), 25.4% (8192). At 8192 the range is 25.4% to 32.0% across the three compressed variants.
- Peak VRAM falls less than the KV cache because weights and activations are shared by all variants. At 8192 GQA-4 saves 75.0% of KV but 25.4% of peak memory (86.0 MiB). MQA saves 108.4 MiB.

### Throughput and latency

![Throughput and Latency](assets/throughput_latency.png)

- Batch-1 decode ranges from 82.7 to 127.2 tok/s (7.86 to 12.09 ms/token) with no consistent ordering. For example, MHA drops to 96.1 tok/s at 2048 and returns to 122.0 at 8192, while MQA falls to 82.7 tok/s at 8192 and GQA-8 reaches 126.0 at the same length. These swings do not track KV size and should be read as noise (Section 6.3).
- Prefill throughput peaks at 2048 (175,418 to 198,303 tok/s) and falls at 8192 (99,364 to 105,818 tok/s).
- MHA has the lowest prefill throughput at 2048, 4096 and 8192. At 2048, MQA reaches 198,303 tok/s against 175,418 for MHA (+13.0%). Each value is a median of 3 runs, so small gaps are not conclusive.

### Full results table

![Summary Table](assets/summary_table.png)

- All 16 (variant, context) rows with every metric are in `results/eval_results.csv` and `summary_table.csv`.
- Perplexity at 2048 is nearly identical across variants (16.87 to 17.12), so divergence only appears at 4096 and beyond.
- At 4096 MHA is best (20.35); GQA-4 and MQA tie at 20.69; GQA-8 is 21.07.

### Pareto analysis

At 8192 the Pareto frontier over (perplexity, peak VRAM) is **{MHA, GQA-4, MQA}**. **GQA-8 is dominated by MQA** at 4096 and 8192 (higher PPL and higher peak VRAM).

**GQA-4 is the recommended operating point among the compressed variants:**

- It has the lowest 8192 perplexity of the three KV-reduced models (33.60 vs 34.27 for MQA and 34.97 for GQA-8).
- It cuts the KV cache by 75.0% (96.75 to 24.19 MiB) and peak VRAM by 25.4% (338.5 to 252.5 MiB).
- MQA saves a further 22.3 MiB of peak VRAM (230.1 vs 252.5) at a cost of 0.67 PPL.
- If long-context quality matters most, MHA remains clearly best at 8192 (30.01 PPL).

---

## 5. Engineering Debugging, Failure Modes and Error Analysis

### 5.1 Why Needle-in-a-Haystack (NIAH) was dropped

A 20M base model trained from scratch on TinyStories is not instruction-tuned. Given a passkey-retrieval prompt, it continues the story instead of extracting the key, which produced a flat 0% retrieval rate even inside $L_{\text{train}}$. A benchmark that scores 0% at every length cannot separate attention variants, so NIAH was removed and perplexity became the quality metric. No NIAH results are reported.

### 5.2 Fixing the `repeat_kv()` memory bottleneck

Early `fp32` runs used `repeat_kv()` to expand K/V back to 16 heads before attention. That materialized full-size K/V tensors in VRAM and masked GQA's memory savings. The fix was native `F.scaled_dot_product_attention(..., enable_gqa=True)` together with `fp16`. With this, peak VRAM reductions at 8k context are real: 25.4% (GQA-4), 29.9% (GQA-8) and 32.0% (MQA). The earlier fp32 runs are not included in this repository's results.

### 5.3 Batch-1 decode throughput noise

At `batch_size=1`, a 20M-parameter `fp16` model (about 40 MB of weights) is dominated by per-step CUDA kernel launch and framework overhead, not memory bandwidth. Decode latency is 7.86 to 12.09 ms/token across all rows. The CSV supports this reading:

- At 8192, MHA reads roughly the whole 96.75 MiB cache each step in about 8.2 ms. That is about 12 GB/s of effective KV traffic, far below the bandwidth of typical GPUs.
- Every variant issues the same number of small kernels per token regardless of KV head count, so a fixed floor hides the bandwidth difference.

No profiler trace is included, so this is an inference from the data. To expose the GQA/MQA bandwidth advantage, decode must be batched (`B >= 16`), which is not benchmarked here.

### Known limitations

- Single training run per variant (seed 42), no confidence intervals. Differences of a few tenths of PPL at 1024 and 2048 should not be over-interpreted.
- Parameter counts differ (20.32M for MHA vs 17.37M to 17.96M), so quality gaps combine the effect of fewer KV heads with fewer parameters.
- Perplexity uses 10 held-out windows per length. Speed uses random tokens, a median of 3 runs, and 64 decode steps.
- `eval_results.csv` does not record the GPU model.

---

## 6. Repository Structure and Reproducibility

```
GQA-Context-Bench/
├── README.md
├── GQA-Context-Bench.ipynb   # Colab runner that writes the .py files below
├── tokenizer.py              # tensor-based byte-level BPE, fast encoder
├── data.py                   # TinyStories loading, content-hashed story split
├── model.py                  # GPT, RMSNorm, RoPE (NTK), SwiGLU, KVCache, attention
├── train.py                  # trains one variant, writes runs/<variant>/
├── evaluate.py               # PPL, VRAM, throughput -> results/eval_results.csv
├── visualize.py              # figures and summary table
├── run_all.py                # train x4 -> evaluate -> visualize
├── results/
│   └── eval_results.csv
├── assets/                   # figures embedded in this README
│   ├── perplexity_curves.png
│   ├── vram_scaling.png
│   ├── throughput_latency.png
│   └── summary_table.png
└── runs/                     # generated: checkpoints and train_log.csv
```

### Setup

```bash
pip install torch datasets pandas matplotlib seaborn   # PyTorch >= 2.5 for enable_gqa
```

### Smoke test (checks the pipeline end to end; results are not meaningful)

```bash
python run_all.py --steps 50 --vocab_size 512 --max_chars 300000 --decode_steps 8
```

### Full benchmark

```bash
python run_all.py \
  --steps 3000 \
  --vocab_size 2048 \
  --block_size 1024 \
  --batch_size 32 \
  --max_chars 20000000 \
  --decode_steps 64
```

### Run stages individually

```bash
python train.py --attention gqa4 --steps 3000 --batch_size 32 --out_dir runs/gqa4   # mha | gqa8 | gqa4 | mqa
python evaluate.py --runs_dir runs --out_csv results/eval_results.csv --max_chars 20000000
python visualize.py --csv results/eval_results.csv --out_dir assets
```

`run_all.py` writes figures to `figures/`. Copy them to `assets/` or run `visualize.py` with `--out_dir assets` as above. Keep `--max_chars` in `evaluate.py` at least as large as in `train.py`, or the validation split will be too short for 8192-token windows.