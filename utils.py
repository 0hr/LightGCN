import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy import sparse
from tqdm.auto import trange, tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_user_items(path):
    user_items = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        user = parts[0]
        items = parts[1:]
        user_items[user].extend(items)
    return user_items


def merge_user_items(train_ui, test_ui):
    combined = defaultdict(list)
    for u, items in train_ui.items():
        combined[u].extend(items)
    for u, items in test_ui.items():
        combined[u].extend(items)
    return combined


def write_user_items(user_items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for user, items in user_items.items():
            if not items:
                continue
            f.write(f"{user} {' '.join(items)}\n")


def build_id_maps(user_items):
    users = sorted(user_items.keys())
    items = sorted({i for lst in user_items.values() for i in lst})
    user_map = {u: idx for idx, u in enumerate(users)}
    item_map = {i: idx for idx, i in enumerate(items)}
    return user_map, item_map


def load_dataset(root, max_users=None, max_items_per_user=None):
    train_path = Path(root) / "train.txt"
    test_path = Path(root) / "test.txt"
    if not train_path.exists():
        raise SystemExit(f"train.txt missing under {root}")

    train_ui = read_user_items(train_path)
    test_ui = read_user_items(test_path) if test_path.exists() else {}

    if max_users is not None:
        if max_users <= 0:
            raise SystemExit("--max-users must be positive")
        keep_users = sorted(train_ui.keys())[:max_users]
        train_ui = {u: train_ui[u] for u in keep_users if train_ui[u]}
        test_ui = {u: items for u, items in test_ui.items() if u in train_ui}

    if max_items_per_user is not None:
        if max_items_per_user <= 0:
            raise SystemExit("--max-items-per-user must be positive")
        trimmed_train = {}
        for u, items in train_ui.items():
            capped = items[:max_items_per_user]
            if capped:
                trimmed_train[u] = capped
        train_ui = trimmed_train
        test_ui = {u: items for u, items in test_ui.items() if u in train_ui}

    if not train_ui:
        raise SystemExit("No training pairs found after applying dataset limits.")

    user_map, item_map = build_id_maps(train_ui)
    train_pairs = []
    for u, items in train_ui.items():
        for i in items:
            train_pairs.append((user_map[u], item_map[i]))

    user_pos = defaultdict(set)
    for u, i in train_pairs:
        user_pos[u].add(i)

    test_pairs = defaultdict(set)
    for u, items in test_ui.items():
        if u in user_map:
            for i in items:
                if i in item_map:
                    test_pairs[user_map[u]].add(item_map[i])

    return len(user_map), len(item_map), train_pairs, user_pos, test_pairs


def build_cv_splits(data_dir, cv_folds, seed):
    train_path = Path(data_dir) / "train.txt"
    test_path = Path(data_dir) / "test.txt"
    train_ui = read_user_items(train_path) if train_path.exists() else {}
    test_ui = read_user_items(test_path) if test_path.exists() else {}
    combined = merge_user_items(train_ui, test_ui)

    if not combined:
        raise SystemExit(f"No interactions found under {data_dir} to create CV folds.")

    rng = np.random.RandomState(seed)
    per_user_parts = {}
    for user, items in combined.items():
        if not items:
            continue
        shuffled = list(items)
        rng.shuffle(shuffled)
        per_user_parts[user] = np.array_split(np.array(shuffled), cv_folds)

    folds = []
    for fold_idx in range(cv_folds):
        train_ui_fold = defaultdict(list)
        test_ui_fold = defaultdict(list)
        for user, parts in per_user_parts.items():
            test_items = parts[fold_idx].tolist()
            train_items = [i for idx, part in enumerate(parts) if idx != fold_idx for i in part.tolist()]
            if train_items:
                train_ui_fold[user] = train_items
                if test_items:
                    test_ui_fold[user] = test_items
        folds.append((train_ui_fold, test_ui_fold))
    return folds


def build_normalized_adj(num_users, num_items, train_pairs, edge_dropout=0.0):
    if len(train_pairs) == 0:
        raise SystemExit("No training pairs found; cannot build graph.")

    pairs = np.asarray(train_pairs, dtype=np.int64).reshape(-1, 2)
    users = pairs[:, 0]
    items = pairs[:, 1]
    if edge_dropout > 0:
        keep = np.random.rand(len(users)) >= edge_dropout
        users = users[keep]
        items = items[keep]

    rows = np.concatenate([users, num_users + items]).astype(np.int64)
    cols = np.concatenate([num_users + items, users]).astype(np.int64)
    data = np.ones_like(rows, dtype=np.float32)

    n_nodes = num_users + num_items
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0.0] = 1.0
    d_root_inv = np.power(deg, -0.5)

    # Symmetric normalization: \tilde{A} = D^{-1/2} A D^{-1/2}
    norm_data = d_root_inv[rows] * data * d_root_inv[cols]
    norm_adj = sparse.coo_matrix((norm_data, (rows, cols)), shape=(n_nodes, n_nodes))
    idx = np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64)
    indices = torch.as_tensor(idx, dtype=torch.long)
    values = torch.as_tensor(norm_adj.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()


def bpr_loss(pos_scores, neg_scores):
    # L_BPR = - E[ log sigma(s(u,i+) - s(u,i-)) ]
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


def recall_ndcg(rank_indices, ground_truth, k: int):
    if not ground_truth:
        return 0.0, 0.0

    if not isinstance(ground_truth, set):
        gt_set = set(ground_truth)
    else:
        gt_set = ground_truth

    hits = 0
    dcg = 0.0

    for rank, idx in enumerate(rank_indices[:k], start=1):
        item_id = int(idx.item())
        if item_id in gt_set:
            hits += 1
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gt_set), k)
    if ideal_hits > 0:
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        ndcg = dcg / idcg
    else:
        ndcg = 0.0

    recall = hits / len(gt_set)
    return recall, ndcg


def evaluate(model, user_embs, item_embs, train_pos, test_pairs, k_list=(20, 50)):
    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    num_items = item_embs.shape[0]
    k_max = min(max(k_list), num_items)

    model.eval()
    with torch.no_grad():
        for u, gt_items in test_pairs.items():
            if not gt_items:
                continue

            gt_set = gt_items if isinstance(gt_items, set) else set(gt_items)

            scores = torch.matmul(item_embs, user_embs[u])

            if u in train_pos and train_pos[u]:
                mask_idx = torch.tensor(list(train_pos[u]), device=scores.device, dtype=torch.long)
                scores[mask_idx] = -1e9

            topk_idx = torch.topk(scores, k_max).indices  # [k_max]

            for k in k_list:
                r, n = recall_ndcg(topk_idx, gt_set, k)
                recalls[k].append(r)
                ndcgs[k].append(n)

    metrics = {
        f"recall@{k}": float(np.mean(recalls[k])) if recalls[k] else 0.0
        for k in k_list
    }
    metrics.update({
        f"ndcg@{k}": float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
        for k in k_list
    })
    return metrics


def evaluate_scores(score_matrix, train_pos, test_pairs, k_list=(20, 50)):
    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    num_users, num_items = score_matrix.shape
    k_max = min(max(k_list), num_items)

    for u, gt_items in test_pairs.items():
        if not gt_items:
            continue

        gt_set = gt_items if isinstance(gt_items, set) else set(gt_items)

        scores = score_matrix[u].clone()

        if u in train_pos and train_pos[u]:
            mask_idx = torch.tensor(list(train_pos[u]), device=scores.device, dtype=torch.long)
            scores[mask_idx] = -1e9

        topk_idx = torch.topk(scores, k_max).indices

        for k in k_list:
            r, n = recall_ndcg(topk_idx, gt_set, k)
            recalls[k].append(r)
            ndcgs[k].append(n)

    metrics = {
        f"recall@{k}": float(np.mean(recalls[k])) if recalls[k] else 0.0
        for k in k_list
    }
    metrics.update({
        f"ndcg@{k}": float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
        for k in k_list
    })
    return metrics


def plot_metric_history(history, model_name, dataset, out_dir):
    if not history:
        return None
    metrics = ["ndcg@20", "ndcg@50", "recall@20", "recall@50"]
    plt.figure(figsize=(8, 5))
    # Only plot points for epochs where the metric exists
    for metric in metrics:
        pts = [(entry["epoch"], entry[metric]) for entry in history if metric in entry]
        if not pts:
            continue
        epochs, values = zip(*pts)
        plt.plot(list(epochs), list(values), label=metric)
    if not plt.gca().has_data():
        plt.close()
        return None
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.title(f"{dataset} - {model_name} metrics")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{dataset}_{model_name}_metrics.png"
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    tqdm.write(f"Saved metrics plot: {file_path}")
    return file_path


def plot_loss_history(history, model_name, dataset, out_dir):
    if not history:
        return None
    # Expect each entry to have 'loss' and 'epoch'
    pts = [(entry["epoch"], entry.get("loss")) for entry in history if "loss" in entry]
    if not pts:
        return None
    epochs, losses = zip(*pts)
    plt.figure(figsize=(8, 4))
    plt.plot(list(epochs), list(losses), label="loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{dataset} - {model_name} loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{dataset}_{model_name}_loss.png"
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    tqdm.write(f"Saved loss plot: {file_path}")
    return file_path


def plot_time_history(history, model_name, dataset, out_dir):
    if not history:
        return None
    epochs = [entry["epoch"] for entry in history]
    times = [entry.get("epoch_time", 0.0) for entry in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, times, label="epoch_time_s")
    plt.xlabel("epoch")
    plt.ylabel("seconds")
    plt.title(f"{dataset} - {model_name} epoch time")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{dataset}_{model_name}_epoch_time.png"
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    tqdm.write(f"Saved epoch time plot: {file_path}")
    return file_path


def format_metrics_table(results):
    metric_keys = set()
    for metrics in results.values():
        metric_keys.update(metrics.keys())
    metric_keys = sorted(metric_keys)
    header = ["model"] + metric_keys

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    rows = []
    for model, metrics in results.items():
        rows.append([model] + [fmt(metrics.get(k, "")) for k in metric_keys])

    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]

    def join(row):
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(header)))

    line = "-+-".join("-" * w for w in widths)
    table = [join(header), line]
    for row in rows:
        table.append(join(row))
    return "\n".join(table)


def plot_time_bar(model_times, out_path):
    if not model_times:
        return None
    names = list(model_times.keys())
    times = [model_times[m] for m in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, times, color="#4c72b0")
    plt.ylabel("train time (s)")
    plt.title("Training time by model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path
