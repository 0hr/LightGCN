import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sampler import Sampler
from utils import  *


# ----------------------------
# Model
# ----------------------------

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=3, learnable_weights=False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # Joint embedding table: [num_users + num_items, d]
        self.embeddings = nn.Embedding(num_users + num_items, embed_dim)
        nn.init.normal_(self.embeddings.weight, std=0.1)

        if learnable_weights:
            self.alpha = nn.Parameter(torch.zeros(num_layers + 1))
        else:
            self.register_buffer("alpha", torch.ones(num_layers + 1) / (num_layers + 1))

    def forward(self, norm_adj):
        x0 = self.embeddings.weight                    # [N, d]
        embs = [x0]
        x = x0
        for _ in range(self.num_layers):
            x = torch.sparse.mm(norm_adj, x)           # E^{k+1} = A_hat E^k
            embs.append(x)

        stacked = torch.stack(embs, dim=1)             # [N, K+1, d]

        if isinstance(self.alpha, nn.Parameter):
            layer_weights = torch.softmax(self.alpha, dim=0)  # [K+1], sums to 1
        else:
            layer_weights = self.alpha

        layer_weights = layer_weights.view(1, -1, 1)    # [1, K+1, 1]
        out = torch.sum(stacked * layer_weights, dim=1) # [N, d]

        user_embs = out[: self.num_users]
        item_embs = out[self.num_users:]
        return user_embs, item_embs

    def score(self, user_embs, item_embs, users, pos_items, neg_items):
        """
        users:       [B] user indices (0..num_users-1)
        pos_items:   [B] item indices (0..num_items-1)
        neg_items:   [B] item indices (0..num_items-1)
        """
        u_vecs = user_embs[users]          # [B, d]
        pos = item_embs[pos_items]         # [B, d]
        neg = item_embs[neg_items]         # [B, d]

        pos_scores = torch.sum(u_vecs * pos, dim=1)
        neg_scores = torch.sum(u_vecs * neg, dim=1)
        return pos_scores, neg_scores


# ----------------------------
# Training helper
# ----------------------------

def run_lightgcn(args, seed, device):
    set_seed(seed)
    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent / "data" / args.dataset
    num_users, num_items, train_pairs, user_pos, test_pairs = load_dataset(
        data_dir, max_users=args.max_users, max_items_per_user=args.max_items_per_user
    )
    tqdm.write(
        f"[LightGCN][Seed {seed}] users={num_users} items={num_items} "
        f"train_edges={len(train_pairs)} test_users={len(test_pairs)}"
    )

    model = LightGCN(
        num_users,
        num_items,
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        learnable_weights=args.learnable_weights,
    ).to(device)

    # If you want paper-style batch L2, set args.use_batch_l2 = True.
    use_batch_l2 = getattr(args, "use_batch_l2", False)

    if use_batch_l2:
        # L2 is handled manually per batch
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    else:
        # Original behavior: BPR + Adam with weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    sampler = Sampler(num_users, num_items, user_pos, neg_strategy=args.neg_sampler)
    base_adj = build_normalized_adj(num_users, num_items, train_pairs, edge_dropout=0.0).to(device)

    best_ndcg = -np.inf
    patience_left = args.patience
    metrics_out = None
    metrics_history = []
    start_time = time.time()

    item_offset = num_users

    for epoch in trange(args.epochs, desc=f"Seed {seed}"):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = min(args.steps_per_epoch, max(1, len(train_pairs) // args.batch_size))

        if args.edge_dropout > 0:
            norm_adj = build_normalized_adj(
                num_users, num_items, train_pairs, edge_dropout=args.edge_dropout
            ).to(device)
        else:
            norm_adj = base_adj

        for _ in range(num_batches):
            users, pos, neg = sampler.sample_batch(args.batch_size)
            users_t = torch.tensor(users, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            optimizer.zero_grad()

            user_embs, item_embs = model(norm_adj)
            pos_scores, neg_scores = model.score(user_embs, item_embs, users_t, pos_t, neg_t)
            bpr = bpr_loss(pos_scores, neg_scores)

            if use_batch_l2:
                emb = model.embeddings.weight
                user_idx = users_t                         # 0..num_users-1
                pos_idx = pos_t + item_offset             # shift items by num_users
                neg_idx = neg_t + item_offset

                reg_loss = 0.5 * (
                    emb[user_idx].pow(2).sum() +
                    emb[pos_idx].pow(2).sum() +
                    emb[neg_idx].pow(2).sum()
                ) / users_t.size(0)                       # average over batch

                loss = bpr + args.l2 * reg_loss
            else:
                loss = bpr

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - t0
        throughput = (num_batches * args.batch_size) / max(epoch_time, 1e-6)
        avg_loss = total_loss / num_batches

        # Prepare history entry with loss for every epoch
        history_entry = {"epoch": int(epoch), "epoch_time": epoch_time, "loss": float(avg_loss)}

        if epoch % args.eval_every == 0 and test_pairs:
            model.eval()
            with torch.no_grad():
                user_embs, item_embs = model(base_adj)
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
                f"Epoch {epoch}: loss={avg_loss:.4f} | metrics={metrics} "
                f"| time={epoch_time:.2f}s | throughput={throughput:.1f}/s"
            )
        else:
            metrics_history.append(history_entry)
            tqdm.write(
                f"Epoch {epoch}: loss={avg_loss:.4f} | time={epoch_time:.2f}s | throughput={throughput:.1f}/s"
            )

    total_time = time.time() - start_time
    return metrics_out or {}, metrics_history, total_time
