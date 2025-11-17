import time
from pathlib import Path
from collections import defaultdict

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

class MultVAE(nn.Module):
    def __init__(self, num_items, hidden=600, latent=64, dropout=0.5):
        super().__init__()
        self.num_items = num_items
        self.latent = latent

        self.enc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_items, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent * 2),
        )
        self.dec = nn.Linear(latent, num_items)

    def forward(self, x):
        """
        x: (batch_size, num_items) implicit feedback vector (multi-hot)
        """
        q = self.enc(x)
        mu, logvar = torch.chunk(q, 2, dim=1)

        if self.training:
            # Stochastic during training
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # Deterministic at eval time (use mean)
            z = mu

        return self.dec(z), mu, logvar

    @staticmethod
    def loss(recon, target, mu, logvar, beta=0.2):
        """
        recon: logits, (B, num_items)
        target: multi-hot implicit vector, (B, num_items)
        mu, logvar: posterior params
        beta: KL weight
        """
        log_softmax_var = F.log_softmax(recon, dim=1)
        recon_loss = -torch.mean(torch.sum(target * log_softmax_var, dim=1))

        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        return recon_loss + beta * kl_loss


# ----------------------------
# Training helper
# ----------------------------

def run_multvae(args, seed, device):
    set_seed(seed)

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent / "data" / args.dataset
    num_users, num_items, train_pairs, user_pos, test_pairs = load_dataset(
        data_dir, max_users=args.max_users, max_items_per_user=args.max_items_per_user
    )

    tqdm.write(
        f"[Mult-VAE][Seed {seed}] users={num_users} items={num_items} "
        f"train_edges={len(train_pairs)} test_users={len(test_pairs)}"
    )

    model = MultVAE(
        num_items,
        hidden=args.vae_hidden,
        latent=args.embed_dim,
        dropout=args.vae_dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    BATCH_LIMIT = 4096
    eff_batch_size = min(args.batch_size, BATCH_LIMIT)

    best_ndcg = -np.inf
    patience_left = args.patience
    metrics_out = None
    metrics_history = []
    start_time = time.time()

    num_batches = min(args.steps_per_epoch, max(1, num_users // eff_batch_size))
    all_users = np.arange(num_users)

    for epoch in trange(args.epochs, desc=f"Seed {seed}"):
        t0 = time.time()
        model.train()
        np.random.shuffle(all_users)
        total_loss = 0.0

        for b in range(num_batches):
            batch_users = all_users[b * eff_batch_size: (b + 1) * eff_batch_size]
            if len(batch_users) == 0:
                continue

            batch_mat = torch.zeros((len(batch_users), num_items), device=device)
            for idx, u in enumerate(batch_users):
                if u in user_pos:
                    items = list(user_pos[u])
                    if items:
                        batch_mat[idx, items] = 1.0

            optimizer.zero_grad()
            recon, mu, logvar = model(batch_mat)
            loss = MultVAE.loss(recon, batch_mat, mu, logvar, beta=args.vae_beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - t0
        throughput = len(train_pairs) / max(epoch_time, 1e-6)
        avg_loss = total_loss / max(num_batches, 1)

        # Record loss for each epoch
        history_entry = {"epoch": int(epoch), "epoch_time": epoch_time, "loss": float(avg_loss)}

        if epoch % args.eval_every == 0 and test_pairs:
            model.eval()
            with torch.no_grad():
                recalls = defaultdict(list)
                ndcgs = defaultdict(list)
                test_user_list = list(test_pairs.keys())
                eval_bs = 4096  # Safe limit for constructing history vectors

                for i in range(0, len(test_user_list), eval_bs):
                    batch_u = test_user_list[i: i + eval_bs]
                    batch_in = torch.zeros((len(batch_u), num_items), device=device)
                    for idx, u in enumerate(batch_u):
                        if u in user_pos:
                            items = list(user_pos[u])
                            if items:
                                batch_in[idx, items] = 1.0

                    recon, _, _ = model(batch_in)
                    # Mask train items
                    recon[batch_in > 0] = -1e9

                    k_max = 50
                    topk = torch.topk(recon, k_max).indices

                    for idx, u in enumerate(batch_u):
                        gt = test_pairs[u]
                        for k in [20, 50]:
                            r, n = recall_ndcg(topk[idx, :k], gt, k)
                            recalls[k].append(r)
                            ndcgs[k].append(n)

            metrics = {f"recall@{k}": float(np.mean(recalls[k])) if recalls[k] else 0.0 for k in [20, 50]}
            metrics.update({f"ndcg@{k}": float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0 for k in [20, 50]})
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
