import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from sampler import Sampler
from utils import *


# ----------------------------
# Model
# ----------------------------

class MFBPR(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        # Standard Gaussian init (BPR-MF style)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self):
        return self.user_emb.weight, self.item_emb.weight

    def score(self, users, pos_items, neg_items):
        u = self.user_emb(users)        # (B, D)
        i = self.item_emb(pos_items)    # (B, D)
        j = self.item_emb(neg_items)    # (B, D)

        pos_scores = (u * i).sum(dim=1)
        neg_scores = (u * j).sum(dim=1)
        return pos_scores, neg_scores


# ----------------------------
# Training helper
# ----------------------------

def run_mf(args, seed, device):
    set_seed(seed)

    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else Path(__file__).resolve().parent.parent / "data" / args.dataset
    )

    num_users, num_items, train_pairs, user_pos, test_pairs = load_dataset(
        data_dir,
        max_users=args.max_users,
        max_items_per_user=args.max_items_per_user,
    )

    tqdm.write(
        f"[MF-BPR][Seed {seed}] users={num_users} "
        f"items={num_items} train_edges={len(train_pairs)} "
        f"test_users={len(test_pairs)}"
    )

    model = MFBPR(num_users, num_items, args.embed_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2
    )

    sampler = Sampler(
        num_users,
        num_items,
        user_pos,
        neg_strategy=args.neg_sampler,
    )

    best_ndcg = -np.inf
    patience_left = args.patience
    metrics_out = None
    metrics_history = []
    start_time = time.time()

    for epoch in trange(args.epochs, desc=f"Seed {seed}"):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        # How many batches per epoch (capped by steps_per_epoch)
        num_batches = min(
            args.steps_per_epoch,
            max(1, len(train_pairs) // args.batch_size),
        )

        for _ in range(num_batches):
            users, pos, neg = sampler.sample_batch(args.batch_size)

            users_t = torch.as_tensor(users, device=device, dtype=torch.long)
            pos_t   = torch.as_tensor(pos,   device=device, dtype=torch.long)
            neg_t   = torch.as_tensor(neg,   device=device, dtype=torch.long)

            optimizer.zero_grad()
            pos_scores, neg_scores = model.score(users_t, pos_t, neg_t)

            # Standard BPR loss: -E[log σ(s_pos - s_neg)]
            loss = bpr_loss(pos_scores, neg_scores)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - t0
        throughput = (num_batches * args.batch_size) / max(epoch_time, 1e-6)
        avg_loss = total_loss / num_batches

        # Prepare history entry including loss for each epoch
        history_entry = {"epoch": int(epoch), "epoch_time": epoch_time, "loss": float(avg_loss)}

        # Evaluation + early stopping
        if epoch % args.eval_every == 0 and len(test_pairs) > 0:
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
                tqdm.write(
                    f"Early stopping at epoch {epoch} | "
                    f"best NDCG@20={best_ndcg:.4f}"
                )
                break

            metrics_history.append(history_entry)
            tqdm.write(
                f"Epoch {epoch}: loss={avg_loss:.4f} | "
                f"metrics={metrics} | time={epoch_time:.2f}s | "
                f"throughput={throughput:.1f}/s"
            )
        else:
            metrics_history.append(history_entry)
            tqdm.write(
                f"Epoch {epoch}: loss={avg_loss:.4f} | "
                f"time={epoch_time:.2f}s | throughput={throughput:.1f}/s"
            )

    total_time = time.time() - start_time
    return metrics_out or {}, metrics_history, total_time
