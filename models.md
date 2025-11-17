Models and Training Details
===========================

This document describes how each model in `models.py` is implemented, the data pipeline, losses, metrics, and the training flow used by the main CLI.

Data pipeline and graph construction
------------------------------------
- Input format: `train.txt` and optional `test.txt` under `data/<dataset>/` (override with `--data-dir`). Each line is `<user> <item1> <item2> ...`.
- Subsetting: `--max-users` keeps only the first N users (after sorting ids); `--max-items-per-user` caps interactions per kept user. Users/items without interactions after subsetting are dropped.
- ID mapping: users and items are remapped to dense 0-based indices for embeddings.
- Graph: all positive `(u, i)` pairs become undirected edges in a bipartite adjacency. The normalized adjacency is
  ```
  A_hat = D^{-1/2} A D^{-1/2}
  ```
  where `D` is the node degree diagonal. Coordinates are built with SciPy COO then converted to a PyTorch sparse tensor.
- Edge dropout: If `--edge-dropout > 0`, a Bernoulli mask drops edges during training graph construction; evaluation always uses the full graph.

Negative sampling (BPR models)
------------------------------
- The sampler picks a random positive item per user and a negative item not seen by that user.
- Strategies:
  - `uniform`: sample negatives uniformly from all items.
  - `popularity`: sample with probability proportional to item frequency.
- If a user has interacted with every item, a random item is returned as a fallback.

Metrics and evaluation
----------------------
- Meanings:
  - Recall@K measures coverage of the held-out relevant items; 1.0 means all ground-truth items appear in the top-K slate.
  - NDCG@K rewards correct items more at higher ranks; it is 1.0 when all relevant items are ranked above non-relevant ones within K.
- Candidate set: all items. Scores for items the user interacted with in training are masked to `-1e9` so they cannot be recommended.
- Top-K retrieval: take `topk(scores, K)` for K in {20, 50}.
- Recall@K:
  ```
  Recall@K = |TopK ∩ GT| / min(|GT|, K)
  ```
  where GT is the set of held-out positives for the user.
- NDCG@K:
  ```
  DCG@K  = sum_{r=1..K} ( rel_r / log2(r + 1) ),  rel_r = 1 if item at rank r is in GT else 0
  IDCG@K = DCG of an ideal ranking with min(|GT|, K) relevant items
  NDCG@K = DCG@K / IDCG@K
  ```
- Aggregation: metrics are averaged across test users; missing GT items yield zeros.
- Cadence: evaluation runs every `--eval-every` epochs. Early stopping monitors NDCG@20 with patience `--patience`.
- Caveats:
  - Multiple ground-truth items per user are supported; Recall/NDCG still normalize by `min(|GT|, K)`.
  - Cold-start users/items (present only in test) are dropped during ID mapping; they do not contribute to metrics.
- Because scores are computed against all items, the metrics reflect a full ranking setting, not sampled evaluation.
- K-fold CV: set `--cv-folds >= 2` (with optional `--cv-seed`) to build user-wise folds from the union of train/test interactions; training/eval runs per fold are averaged.

Training loop (shared BPR structure)
------------------------------------
1. Build (or reuse) normalized adjacency; optionally rebuild per epoch if edge dropout is on.
2. For each mini-batch (size `--batch-size`, capped by `--steps-per-epoch`):
   - Sample `(user, pos, neg)` triples.
   - Obtain model-specific user/item embeddings.
   - Compute scores and the BPR loss:
     ```
     L_BPR = -E[ log sigma(s(u,i_pos) - s(u,i_neg)) ]
     ```
   - Optimize with Adam (lr `--lr`, weight decay `--l2`).
3. Log loss, throughput, and eval metrics. Metrics/history are later plotted per model; training-time bars are aggregated across models.

Model summaries, references, pros/cons
--------------------------------------
- LightGCN — Simplifying GCNs for recommendation by removing feature transforms and nonlinearities; averages embeddings across propagation layers.
  - Paper: https://arxiv.org/abs/2002.02126
  - Pros: very lightweight per layer, strong accuracy on implicit CF; layer weighting optional; stable training.
  - Cons: relies on clean graph structure; no feature fusion beyond connectivity; edge weights are uniform.
- xGCN — Lightweight baseline that keeps only the final propagated embedding (no layer aggregation). Used here as a pared-down sanity check.
  - Pros: fastest among graph models; minimal overhead.
  - Cons: less expressive than LightGCN aggregation; sensitive to layer count.
- NGCF — Neural Graph Collaborative Filtering with learned transforms and nonlinearities.
  - Paper: https://arxiv.org/abs/1905.08108
  - Pros: greater expressiveness via per-layer weights and activations.
  - Cons: heavier compute; risk of over-smoothing/overfitting on sparse graphs.
- MF-BPR — Matrix factorization trained with Bayesian Personalized Ranking (pairwise logistic loss).
  - Paper: https://arxiv.org/abs/1205.2618 (BPR)
  - Pros: simple, fast, strong baseline; no graph construction required.
  - Cons: ignores higher-order connectivity; limited to 1-hop signal.
- WMF — Weighted matrix factorization for implicit feedback with confidence weights.
  - Paper: https://doi.org/10.1109/ICDM.2008.22 (Hu, Koren, Volinsky 2008/2010)
  - Pros: uses all observations each epoch; principled confidence weighting.
  - Cons: dense matrix memory cost; full-matrix updates per epoch are heavier than sampling.
- Mult-VAE — Variational autoencoder for implicit CF with multinomial likelihood.
  - Paper: https://arxiv.org/abs/1802.05814 (Liang et al., 2018)
  - Pros: captures nonlinear user preference structure; calibrated via KL term.
- Cons: dense input/output per batch; KL tuning (`beta`) required; decoder logits over all items.

Quick compute/behavior comparison (qualitative)
-----------------------------------------------
| Model    | Prop steps | Loss type | Relative cost | Notes |
| ---      | ---        | ---       | ---           | ---   |
| LightGCN | K sparse matmuls + layer avg | Pairwise BPR | Low | Strong accuracy vs. compute; optional learnable alphas. |
| xGCN     | K sparse matmuls | Pairwise BPR | Lowest | No layer fusion; fastest graph baseline. |
| NGCF     | K sparse matmuls + linear layers + LeakyReLU | Pairwise BPR | Medium-High | More expressive; higher compute. |
| MF-BPR   | None (no graph) | Pairwise BPR | Low | Embedding-only; ignores graph structure. |
| WMF      | None (dense full pass) | Weighted MSE | Medium | Dense matrix per epoch; uses all data. |
| Mult-VAE | None (dense) | Recon + KL | High | Dense encoder/decoder over all items per batch. |

LightGCN
--------
- Node embeddings: one trainable matrix `E ∈ R^{(U+I)×d}` initialized N(0, 0.01).
- Propagation for hop `k`:
  ```
  E^{(0)} = E
  E^{(k+1)} = A_hat * E^{(k)}
  ```
  where `A_hat` is symmetric normalized adjacency.
- Clustering of layers:
  - Without `--learnable-weights`: `alpha_k = 1 / (K + 1)` for k ∈ [0, K].
  - With `--learnable-weights`: `alpha` is a learnable vector passed through softmax at runtime.
- Final embeddings:
  ```
  E_out = sum_{k=0..K} alpha_k * E^{(k)}
  U = E_out[0:U],  V = E_out[U:U+I]
  ```
- Scoring: `s(u, i) = <U_u, V_i>`. Loss: shared BPR logistic loss.

XGCN (lightweight GCN baseline)
-------------------------------
- Same `E` and `A_hat` construction as LightGCN.
- Propagation: apply `A_hat` K times; discard intermediate layers and keep only `E^{(K)}`.
- Output split: `U = E^{(K)}_users`, `V = E^{(K)}_items`.
- Scoring and loss: dot product + BPR, identical masking/eval.
- Constraint: `--layers >= 1`; otherwise the script exits.

NGCF (simplified)
-----------------
- Parameters: per-layer weight matrices `W_l ∈ R^{d×d}`, initialized Xavier.
- Message passing (without the full NGCF gating/concat terms):
  ```
  H^{(0)} = E
  Z^{(l)} = A_hat * H^{(l)}             # neighbor aggregation
  H^{(l+1)} = LeakyReLU( W_l Z^{(l)} )  # learned transform + nonlinearity
  ```
- Layer fusion: arithmetic mean of all `H^{(l)}` for l ∈ [0, K].
- Output split into users/items; scoring via dot product with BPR.

MF-BPR
------
- Parameters: user embeddings `P ∈ R^{U×d}`, item embeddings `Q ∈ R^{I×d}`, initialized N(0, 0.01).
- Score: `s(u, i) = <P_u, Q_i>`.
- Loss: same BPR logistic objective with sampled negatives; no graph propagation.

WMF (weighted matrix factorization)
-----------------------------------
- Binary implicit matrix `R ∈ {0,1}^{U×I}` built dense on device from training pairs.
- Confidence matrix: `C = 1 + alpha * R`, where `alpha = --wmf-alpha`.
- Predictions: `R_hat = P Q^T` with trainable user/item embeddings `P, Q`.
- Loss:
  ```
  L = mean( C ⊙ (R_hat - R)^2 )
  ```
- Optimization: one full-matrix pass per epoch with Adam, weight decay `--l2`; no sampling.
- Evaluation: dot-product scores with train-mask treatment identical to BPR models.

Mult-VAE (implicit)
-------------------
- Input: dense binary rows of `R` for a batch of users.
- Encoder:
  ```
  H1 = Dropout(R, p=--vae-dropout)
  H2 = Tanh( W1 H1 + b1 ),  W1 ∈ R^{hidden×I}
  Q  = W2 H2 + b2,          W2 ∈ R^{2*latent×hidden}
  [mu, logvar] = split(Q)
  ```
- Reparameterization: `z = mu + eps * exp(0.5 * logvar)`, `eps ~ N(0, I)`.
- Decoder: `logits = W_dec z + b_dec`, `W_dec ∈ R^{I×latent}`.
- Loss per batch:
  ```
  recon = -sum( R * log_softmax(logits) )          # multinomial CE for implicit counts
  kl    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
  L     = mean(recon) + beta * kl,  beta = --vae-beta
  ```
- Batching: users are shuffled; up to `--steps-per-epoch` batches of size `--batch-size`.
- Evaluation: forward pass on the full user-item matrix to obtain per-user logits; apply train masking and top-K metrics.

Outputs and logging
-------------------
- Per-model console: epoch loss, throughput (samples/sec), optional eval metrics, early-stopping notices, and CUDA memory peak (when available and not `--cpu`).
- Plots (`--plot-dir`):
  - `<dataset>_<model>_metrics.png`: curves for NDCG@20/50 and Recall@20/50 across epochs.
  - `<dataset>_<model>_epoch_time.png`: wall time per epoch.
- Aggregated artifacts:
  - Results table printed at the end with metrics per model and `train_time_s`.
  - `<dataset>_train_times.png`: bar chart comparing average train time across run models.
- Multi-seed runs: metrics and train times are averaged over `--seeds`; individual histories still drive the per-seed plots.
