import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sampler import Sampler
from utils import *


# ----------------------------
# Model
# ----------------------------

class WMF(nn.Module):
    """
    Implicit Weighted Matrix Factorization

    - binary interactions: p_ui in {0, 1}
    - confidence: c_ui = 1 + alpha * p_ui
    - objective: sum_{u,i} c_ui * (p_ui - r_ui)^2
    """
    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self):
        # Return full embedding matrices (U, V)
        return self.user_emb.weight, self.item_emb.weight

    def loss(self, user_indices, item_vecs, batch_targets, alpha=1.0):
        """
        user_indices : (B,)
        item_vecs    : (num_items, D)
        batch_targets: (B, num_items), entries in {0,1}
        alpha        : confidence scaling for positives
        """
        u_vecs = self.user_emb(user_indices)          # (B, D)
        preds = u_vecs @ item_vecs.T                  # (B, num_items)

        # WMF confidence: c_ui = 1 + alpha * p_ui
        confidence = 1.0 + alpha * batch_targets      # (B, num_items)

        diff = preds - batch_targets
        loss = (confidence * diff.pow(2)).mean()
        return loss


# ----------------------------
# Training helper
# ----------------------------

def run_wmf(args, seed, device):
    set_seed(seed)

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent / "data" / args.dataset
    num_users, num_items, train_pairs, user_pos, test_pairs = load_dataset(
        data_dir,
        max_users=args.max_users,
        max_items_per_user=args.max_items_per_user,
    )

    tqdm.write(
        f"[WMF][Seed {seed}] users={num_users} "
        f"items={num_items} train_edges={len(train_pairs)} "
        f"test_users={len(test_pairs)}"
    )

    model = WMF(num_users, num_items, args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    BATCH_LIMIT = 4096
    eff_batch_size = min(args.batch_size, BATCH_LIMIT)

    best_ndcg = -np.inf
    patience_left = args.patience
    metrics_out = None
    metrics_history = []
    start_time = time.time()

    all_users = np.arange(num_users)

    max_batches = math.ceil(num_users / eff_batch_size)
    num_batches = min(args.steps_per_epoch, max_batches)

    for epoch in trange(args.epochs, desc=f"Seed {seed}"):
        t0 = time.time()
        model.train()
        np.random.shuffle(all_users)

        total_loss = 0.0

        for b in range(num_batches):
            start = b * eff_batch_size
            end = min((b + 1) * eff_batch_size, num_users)
            batch_users = all_users[start:end]
            if len(batch_users) == 0:
                continue

            # Build dense implicit feedback matrix for this batch
            batch_target = torch.zeros((len(batch_users), num_items), device=device)
            for idx, u in enumerate(batch_users):
                if u in user_pos:
                    items = list(user_pos[u])
                    if items:
                        batch_target[idx, items] = 1.0

            batch_users_t = torch.tensor(batch_users, dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss = model.loss(
                user_indices=batch_users_t,
                item_vecs=model.item_emb.weight,
                batch_targets=batch_target,
                alpha=args.wmf_alpha,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - t0
        throughput = len(train_pairs) / max(epoch_time, 1e-6)
        avg_loss = total_loss / max(1, num_batches)

        # Record loss every epoch
        history_entry = {"epoch": int(epoch), "epoch_time": epoch_time, "loss": float(avg_loss)}

        if epoch % args.eval_every == 0 and test_pairs:
            model.eval()
            with torch.no_grad():
                user_embs, item_embs = model.forward()
            metrics = evaluate(model, user_embs, item_embs, user_pos, test_pairs)
            history_entry.update(metrics)

            ndcg20 = metrics.get("ndcg@20", 0.0)
            if ndcg20 > best_ndcg:
                best_ndcg = ndcg20
                patience_left = args.patience
                metrics_out = metrics
            else:
                patience_left -= 1

            if patience_left <= 0:
                tqdm.write(f"Early stopping at epoch {epoch} | best NDCG@20={best_ndcg:.4f}")
                break

            metrics_history.append(history_entry)
            tqdm.write(
                f"Epoch {epoch}: loss={avg_loss:.4f} | metrics={metrics} | "
                f"time={epoch_time:.2f}s | throughput={throughput:.1f}/s"
            )
        else:
            metrics_history.append(history_entry)
            tqdm.write(
                f"Epoch {epoch}: loss={avg_loss:.4f} | "
                f"time={epoch_time:.2f}s | throughput={throughput:.1f}/s"
            )

    total_time = time.time() - start_time
    return metrics_out or {}, metrics_history, total_time
