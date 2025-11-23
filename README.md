## Reimplementation and Analysis of LightGCN for Collaborative Filtering


- Models: LightGCN, NGCF, MF-BPR, weighted MF, Mult-VAE.
- Evaluation: Recall@20/50 and NDCG@20/50 on leave-one-out splits; per-epoch histories and plots.


#### To Create virtual environment 

```
python -m venv .venv
```
#### For Macos/Linux

```
source .venv/bin/active
```

#### For Windows

```
.\venv\Scripts\Activate.ps1
```

#### To Install packages
```
pip install -r requirements.txt
```
#### To Download datasets
```
python data.py --<dataset name>
```

```
python preprocessing.py --dataset <dataset name>
```

#### To Run Models
```
python run.py --dataset amazon --model lightgcn --epochs 50 --eval-every 5
```


Run every model
```
python run.py --dataset amazon --model all --max-users 200 --max-items-per-user 50 \
  --epochs 50 --eval-every 5 --plot-dir plots
```

Dataset layout
- Place `train.txt` and `test.txt` under `./data/<dataset>/` (or pass `--data-dir`).
- Files are whitespace-separated: first token is user id, rest are item ids.

Training & evaluation (`run.py`)
- Outputs: per-epoch logs, metric curves `<dataset>_<model>_metrics.png`, epoch-time plots `<dataset>_<model>_epoch_time.png`, combined results table, and a training-time bar chart `<dataset>_train_times.png` in `--plot-dir`.
- Example: `python run.py --dataset gowalla --model lightgcn --layers 3 --edge-dropout 0.1 --neg-sampler popularity --epochs 100`

| Flag                                               | Default | Notes |
|----------------------------------------------------| --- | --- |
| `--dataset {amazon,gowalla,yelp}`                  | required | Dataset folder under `data/` (use `--data-dir` to override). |
| `--data-dir PATH`                                  | `./data/<dataset>` | Custom dataset path containing `train.txt`/`test.txt`. |
| `--model {lightgcn, ngcf,mf-bpr,wmf,mult-vae,all}` | `all` | Leave unset or use `all` to run every model in order. |
| `--embed-dim INT`                                  | `64` | Embedding / latent dimension. |
| `--layers INT`                                     | `3` | Propagation layers (LightGCN/NGCF). |
| `--learnable-weights`                              | off | LightGCN only; layer weights become learnable (softmax-normalized). |
| `--edge-dropout FLOAT`                             | `0.0` | Edge dropout during training; evaluation always uses the full graph. |
| `--neg-sampler {uniform,popularity}`               | `uniform` | BPR negative sampling strategy. |
| `--batch-size INT`                                 | `512` | Mini-batch size. |
| `--steps-per-epoch INT`                            | `1024` | Cap on training steps per epoch (controls compute cost). |
| `--epochs INT`                                     | `200` | Training epochs. |
| `--lr FLOAT`                                       | `1e-3` | Learning rate. |
| `--l2 FLOAT`                                       | `1e-4` | Weight decay. |
| `--eval-every INT`                                 | `1` | Evaluate every N epochs. |
| `--patience INT`                                   | `20` | Early stopping patience on NDCG@20. |
| `--cpu`                                            | off | Force CPU even if CUDA is available. |
| `--seed INT`                                       | `42` | Primary seed. |
| `--seeds LIST`                                     | unset | Optional list of seeds to average metrics and train time. |
| `--cv-folds INT`                                   | `1` | User-wise K-fold CV when set to ≥2; averages metrics across folds. |
| `--cv-seed INT`                                    | unset | Seed for CV shuffling (defaults to `--seed`). |
| `--max-users INT`                                  | unset | Limit users loaded for faster sweeps. |
| `--max-items-per-user INT`                         | unset | Cap interactions per user when subsetting. |
| `--plot-dir PATH`                                  | `plots/` | Where to save plots and aggregated comparisons. |
| WMF-only: `--wmf-alpha FLOAT`                      | `1.0` | Confidence scaling for weighted MF. |
| VAE-only: `--vae-hidden INT`                       | `600` | Hidden layer size for encoder. |
| VAE-only: `--vae-beta FLOAT`                       | `0.2` | Beta multiplier on KL term. |
| VAE-only: `--vae-dropout FLOAT`                    | `0.5` | Dropout rate for VAE encoder. |

Dataset downloader (`data.py`)
- Usage examples: `python data.py --amazon`, `python data.py --dataset yelp --force`, `python data.py --list`.
- Parameters:
  - `--dataset {amazon,gowalla,yelp}` or shortcuts `--amazon/--gowalla/--yelp`: dataset to download into `./data/<dataset>`.
  - `--base-url URL` (default: upstream LightGCN repo): override download base.
  - `--force`: re-download even if files exist.
  - `--list`: list supported datasets and exit.

Dataset downloader (`preprocessing.py`)
- Usage examples: `python preprocessing.py --amazon`, `python preprocessing.py --dataset yelp --force`, `python data.py --list`.
- Parameters:
  - `--dataset {amazon,gowalla,yelp, ml-1m}` : dataset to download and process.
  - `--raw-path PATH` Path to raw file for Yelp (or Amazon if you have it locally).
  - `--core`: K-core filter size.
  - `--split`: loo (1 item test) or ratio (20% test).

Visualizations
- Bipartite subgraph (`bipartite_graph.py`): `python bipartite_graph.py gowalla --users 20 --interactions 15 --save graph.png`
  - Positional `dataset {amazon,gowalla,yelp}`; `--input PATH` custom train/test file; `--users INT` (default 10); `--interactions INT` min edges per shown user (default 10); `--save PATH` to write the plot.
- Interaction heatmap (`heatmap.py`): `python heatmap.py amazon --users 40 --items 80 --save heatmap.png`
  - Positional `dataset {amazon,gowalla,yelp}`; `--input PATH` override dataset file; `--users INT` max users (default 30); `--items INT` max items (default 50); `--save PATH` to write the image instead of showing it.

Team Members:
Ferit Ozdaban (A20571925)
Harun Rasit Pekacar (A20607262)

This repository contains our final project for CS584, where we reimplement and analyze the LightGCN model from the SIGIR 2020 paper “LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.” LightGCN simplifies graph-based recommendation by removing nonlinearities and feature transformations, while still leveraging high-order user–item connections.

Our project goals:
• Reimplementing LightGCN from scratch in PyTorch
• Reproducing key results from the original paper
• Building and tuning strong baselines (MF-BPR, NGCF, WMF, Mult-VAE)
• Conducting extensive ablations (embedding size, K layers, regularization, sampling, layer weights, dropout)
• Profiling the model’s efficiency and scalability
• Designing at least one principled extension to LightGCN
