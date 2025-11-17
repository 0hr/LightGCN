#!/usr/bin/env python3
"""Visualize sampled bipartite subgraphs for LightGCN datasets."""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import dataset

DATASET_DEFAULTS = {
    "amazon": {
        "path": Path(__file__).resolve().parent / "data" / "amazon" / "train.txt",
        "users": 10,
        "interactions": 10,
    },
    "gowalla": {
        "path": Path(__file__).resolve().parent / "data" / "gowalla" / "train.txt",
        "users": 10,
        "interactions": 10,
    },
    "yelp": {
        "path": Path(__file__).resolve().parent / "data" / "yelp" / "train.txt",
        "users": 10,
        "interactions": 10,
    },
}


def parse_edges(path, max_users=10, interactions_per_user=10):
    edges = []
    users_seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = f"u_{parts[0]}"

            if len(users_seen) >= max_users and user not in users_seen:
                continue

            items = parts[1: interactions_per_user + 1]
            if not items:
                continue

            users_seen.add(user)
            for item in items:
                edges.append((user, f"i_{item}"))

            if len(users_seen) >= max_users and _has_min_edges(users_seen, edges, interactions_per_user):
                break
    return edges


def _has_min_edges(users_seen, edges, interactions_per_user):
    counts = {u: 0 for u in users_seen}
    for u, _ in edges:
        if u in counts:
            counts[u] += 1
    return all(counts[u] >= interactions_per_user for u in counts)


def build_graph(edges):
    graph = nx.Graph()
    users = {u for u, _ in edges}
    items = {i for _, i in edges}
    graph.add_nodes_from(users, bipartite=0)
    graph.add_nodes_from(items, bipartite=1)
    graph.add_edges_from(edges)
    return graph, users, items


def plot_graph(graph, users, output=None, title=None):
    pos = nx.bipartite_layout(graph, users)
    colors = ["#1f77b4" if node in users else "#ff7f0e" for node in graph.nodes]
    sizes = [200 if node in users else 120 for node in graph.nodes]

    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        pos,
        node_color=colors,
        node_size=sizes,
        with_labels=True,
        labels={node: node for node in graph.nodes},
        font_size=6,
        edge_color="#bbbbbb",
    )
    plt.title(title or "Sampled bipartite subgraph")
    user_patch = mpatches.Patch(color="#1f77b4", label="Users (u_<id>)")
    item_patch = mpatches.Patch(color="#ff7f0e", label="Items (i_<id>)")
    plt.legend(handles=[user_patch, item_patch], loc="upper right")
    plt.figtext(0.02, 0.02, "Each edge is a user-item interaction from the sample.", fontsize=9, ha="left")
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=200)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def derive_defaults(dataset):
    defaults = DATASET_DEFAULTS.get(dataset, {})
    return (
        defaults.get("path"),
        defaults.get("users", 10),
        defaults.get("interactions", 10),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Show a sampled bipartite graph from LightGCN datasets (amazon, gowalla, yelp)."
    )
    parser.add_argument("dataset", choices=sorted(DATASET_DEFAULTS.keys()), default="amazon", help="Dataset to visualize")
    parser.add_argument("--input", help="Path to a train/test file (overrides dataset default)", default=DATASET_DEFAULTS['amazon']["path"])
    parser.add_argument("--users", type=int, help="Number of users to include", default=10)
    parser.add_argument("--interactions", type=int, help="Minimum interactions per shown user", default=10)
    parser.add_argument("--save", default=None, help="Optional path to save the plot instead of showing it")
    args = parser.parse_args()

    default_path, default_users, default_interactions = derive_defaults(args.dataset)
    data_path = Path(args.input) if args.input else default_path
    user_count = args.users if args.users is not None else default_users
    interaction_count = args.interactions if args.interactions is not None else default_interactions

    if not data_path or not Path(data_path).exists():
        raise SystemExit(f"Input file not found: {data_path}")

    edges = parse_edges(data_path, max_users=user_count, interactions_per_user=interaction_count)
    if not edges:
        raise SystemExit("No edges were parsed from the input file. Please check the dataset format.")

    graph, users, _ = build_graph(edges)
    title = f"{args.dataset.title()} interactions (sampled bipartite subgraph)"
    plot_graph(graph, users, output=args.save, title=title)


if __name__ == "__main__":
    main()
