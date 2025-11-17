import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sampler import Sampler
from utils import  *

# ----------------------------
# Model
# ----------------------------

class NGCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # user + item embeddings (stacked)
        self.embeddings = nn.Embedding(num_users + num_items, embed_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

        # NGCF transformation weights
        self.W1 = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=True) for _ in range(num_layers)]
        )
        self.W2 = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=True) for _ in range(num_layers)]
        )

        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, norm_adj: torch.sparse.FloatTensor):
        """
        norm_adj: (num_users + num_items, num_users + num_items) sparse normalized adjacency
                  typically constructed on the bipartite graph (no self-loops),
                  so that side_emb = sum_{i in N(u)} e_i / c_{ui}.

        Propagation (per layer):
          side = A_hat e
          sum  = e + side
          bi   = e * side
          e'   = LeakyReLU( W1 * sum + W2 * bi )
        """
        x = self.embeddings.weight  # (U + I, d)
        final_embs = [x]

        for w1, w2 in zip(self.W1, self.W2):
            # Neighbor aggregation
            side_emb = torch.sparse.mm(norm_adj, x)  # (U + I, d)

            # Linear and bi-linear terms
            sum_emb = x + side_emb          # self + neighbors
            bi_emb = x * side_emb           # element-wise interaction

            # NGCF update
            x = self.leaky(w1(sum_emb) + w2(bi_emb))
            x = self.dropout(x)
            x = F.normalize(x, p=2, dim=1)  # optional L2 norm as in many NGCF impls

            final_embs.append(x)

        # Concatenate layer-wise embeddings (0..L)
        out = torch.cat(final_embs, dim=1)  # (U + I, d * (L + 1))

        user_embs = out[: self.num_users]
        item_embs = out[self.num_users :]
        return user_embs, item_embs

    def score(self, user_embs, item_embs, users, pos_items, neg_items):
        """
        Dot-product BPR scoring.
        """
        u_vecs = user_embs[users]          # (B, D)
        pos = item_embs[pos_items]         # (B, D)
        neg = item_embs[neg_items]         # (B, D)
        pos_scores = torch.sum(u_vecs * pos, dim=1)
        neg_scores = torch.sum(u_vecs * neg, dim=1)
        return pos_scores, neg_scores

# ----------------------------
# Training helper
# ----------------------------

def run_ngcf(args, seed, device):
    set_seed(seed)
    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent / "data" / args.dataset
    num_users, num_items, train_pairs, user_pos, test_pairs = load_dataset(
        data_dir, max_users=args.max_users, max_items_per_user=args.max_items_per_user
    )
    tqdm.write(f"[NGCF][Seed {seed}] users={num_users} items={num_items} train_edges={len(train_pairs)} test_users={len(test_pairs)}")
    model = NGCF(num_users, num_items, args.embed_dim, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    sampler = Sampler(num_users, num_items, user_pos, neg_strategy=args.neg_sampler)
    base_adj = build_normalized_adj(num_users, num_items, train_pairs, edge_dropout=0.0).to(device)

    best_ndcg = -np.inf
    patience_left = args.patience
    metrics_out = None
    metrics_history = []
    start_time = time.time()

    for epoch in trange(args.epochs, desc=f"Seed {seed}"):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = min(args.steps_per_epoch, max(1, len(train_pairs) // args.batch_size))
        if args.edge_dropout > 0:
            norm_adj = build_normalized_adj(num_users, num_items, train_pairs, edge_dropout=args.edge_dropout).to(device)
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
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - t0
        throughput = (num_batches * args.batch_size) / max(epoch_time, 1e-6)
        avg_loss = total_loss / num_batches

        # Record loss for every epoch
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
            tqdm.write(f"Epoch {epoch}: loss={avg_loss:.4f} | metrics={metrics} | time={epoch_time:.2f}s | throughput={throughput:.1f}/s")
        else:
            metrics_history.append(history_entry)
            tqdm.write(f"Epoch {epoch}: loss={avg_loss:.4f} | time={epoch_time:.2f}s | throughput={throughput:.1f}/s")

    total_time = time.time() - start_time
    return metrics_out or {}, metrics_history, total_time
