#!/usr/bin/env python3
"""Plot a user-item interaction heatmap for LightGCN datasets."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASET_DEFAULTS = {
    "amazon": Path(__file__).resolve().parent / "data" / "amazon" / "train.txt",
    "gowalla": Path(__file__).resolve().parent / "data" / "gowalla" / "train.txt",
    "yelp": Path(__file__).resolve().parent / "data" / "yelp" / "train.txt",
}


def parse_interactions(path, max_users=30, max_items=50):
    users = []
    items = []
    item_index = {}
    user_index = {}
    pairs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = f"u_{parts[0]}"
            if user not in user_index:
                if len(users) >= max_users:
                    continue
                user_index[user] = len(users)
                users.append(user)

            # take items until cap is reached
            for item_id in parts[1:]:
                item = f"i_{item_id}"
                if item not in item_index:
                    if len(items) >= max_items:
                        continue
                    item_index[item] = len(items)
                    items.append(item)
                # only store if both user/item within caps
                pairs.append((user_index[user], item_index[item]))

            if len(users) >= max_users and len(items) >= max_items:
                break

    return users, items, pairs


def build_matrix(users, items, pairs):
    matrix = np.zeros((len(users), len(items)), dtype=int)
    for u_idx, i_idx in pairs:
        if u_idx < len(users) and i_idx < len(items):
            matrix[u_idx, i_idx] = 1
    return matrix


def plot_heatmap(matrix, users, items, output=None, title=None):
    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap("Blues")
    plt.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Interaction (1=present)")

    # Show every nth label to keep it readable when there are many items/users.
    max_user_labels = 30
    max_item_labels = 40
    user_step = max(1, len(users) // max_user_labels or 1)
    item_step = max(1, len(items) // max_item_labels or 1)

    plt.yticks(range(0, len(users), user_step), users[0::user_step])
    plt.xticks(range(0, len(items), item_step), items[0::item_step], rotation=90)

    plt.xlabel("Items")
    plt.ylabel("Users")
    plt.title(title or "User-Item Interaction Heatmap")
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=200)
        print(f"Saved heatmap to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot a user-item heatmap for LightGCN datasets")
    parser.add_argument("dataset", choices=sorted(DATASET_DEFAULTS.keys()), help="Dataset to visualize")
    parser.add_argument("--input", help="Path to a train/test file (overrides dataset default)")
    parser.add_argument("--users", type=int, default=100, help="Max users to include (default: 30)")
    parser.add_argument("--items", type=int, default=50, help="Max items to include (default: 50)")
    parser.add_argument("--save", default=None, help="Optional path to save the heatmap instead of showing it")
    args = parser.parse_args()

    data_path = Path(args.input) if args.input else DATASET_DEFAULTS[args.dataset]
    if not data_path.exists():
        raise SystemExit(f"Input file not found: {data_path}")

    users, items, pairs = parse_interactions(data_path, max_users=args.users, max_items=args.items)
    if not users or not items:
        raise SystemExit("No users or items parsed; please check the dataset format or adjust limits.")

    matrix = build_matrix(users, items, pairs)
    title = f"{args.dataset.title()} user-item heatmap (users={len(users)}, items={len(items)})"
    plot_heatmap(matrix, users, items, output=args.save, title=title)


if __name__ == "__main__":
    main()
