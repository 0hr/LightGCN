import argparse
import pandas as pd
import numpy as np
import requests
import zipfile
import gzip
import shutil
import os
from pathlib import Path
from tqdm.auto import tqdm


DATA_DIR = Path("./data")
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
GOWALLA_URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
AMAZON_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv"


# ---------------------------------------------------------
# Download Helpers
# ---------------------------------------------------------

def download_file(url, dest_path):
    if dest_path.exists():
        tqdm.write(f"File already exists: {dest_path}, skipping download.")
        return

    tqdm.write(f"Downloading {url}...")
    tqdm.write("This might take a while depending on file size...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        total_size = int(response.headers.get('content-length', 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
            for data in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                f.write(data)
                bar.update(len(data))
    except Exception as e:
        tqdm.write(f"Download failed: {e}")
        if dest_path.exists():
            os.remove(dest_path)  # Clean up partial file
        raise


def extract_zip(zip_path, extract_to):
    tqdm.write(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def extract_gzip(gzip_path, dest_path):
    tqdm.write(f"Extracting {gzip_path}...")
    with gzip.open(gzip_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


# ---------------------------------------------------------
# Core Processing Logic
# ---------------------------------------------------------

def iterative_k_core(df, user_col='user', item_col='item', k=10):
    """
    Recursively filters data until all users and items have at least K interactions.
    """
    tqdm.write(f"Starting Iterative {k}-Core filtering...")
    iteration = 0
    while True:
        n_users_before = df[user_col].nunique()
        n_items_before = df[item_col].nunique()

        # Filter users
        user_counts = df[user_col].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df[user_col].isin(valid_users)]

        # Filter items
        item_counts = df[item_col].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df[item_col].isin(valid_items)]

        n_users_after = df[user_col].nunique()
        n_items_after = df[item_col].nunique()

        tqdm.write(f"  Iter {iteration}: Users {n_users_before}->{n_users_after}, Items {n_items_before}->{n_items_after}")

        if n_users_before == n_users_after and n_items_before == n_items_after:
            break
        iteration += 1

    return df.copy()


def remap_ids(df, user_col, item_col, out_dir):
    """
    Maps user and item IDs to continuous integers 0..N-1 and 0..M-1.
    Saves the mapping to user_list.txt and item_list.txt.
    """
    tqdm.write("Remapping IDs and generating map files...")
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_users = sorted(df[user_col].unique())
    unique_items = sorted(df[item_col].unique())

    user_map = {u: i for i, u in enumerate(unique_users)}
    item_map = {item: i for i, item in enumerate(unique_items)}

    df['user_idx'] = df[user_col].map(user_map).astype('int32')
    df['item_idx'] = df[item_col].map(item_map).astype('int32')

    # Save User List (org_id remap_id)
    user_list_path = out_dir / "user_list.txt"
    tqdm.write(f"  Saving {user_list_path}...")
    with open(user_list_path, 'w') as f:
        f.write("org_id remap_id\n")
        for org_id, remap_id in user_map.items():
            f.write(f"{org_id} {remap_id}\n")

    # Save Item List (org_id remap_id)
    item_list_path = out_dir / "item_list.txt"
    tqdm.write(f"  Saving {item_list_path}...")
    with open(item_list_path, 'w') as f:
        f.write("org_id remap_id\n")
        for org_id, remap_id in item_map.items():
            f.write(f"{org_id} {remap_id}\n")

    tqdm.write(f"Final Stats: {len(user_map)} users, {len(item_map)} items, {len(df)} interactions.")
    return df


def robust_loo_split(df, user_col='user_idx', time_col='time'):
    """
    Strict Leave-One-Out split (Last item to test).
    """
    tqdm.write("Performing Robust Leave-One-Out Split...")

    # 1. Sort: Primary=User, Secondary=Time, Tertiary=Item (deterministic tie-break)
    df = df.sort_values(by=[user_col, time_col, 'item_idx'])

    # 2. Identify the last item per user
    train_mask = df.duplicated(subset=[user_col], keep='last')

    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    # 3. Ensure Test users exist in Train
    train_users = set(train_df[user_col].unique())
    test_df = test_df[test_df[user_col].isin(train_users)]

    n_users = test_df[user_col].nunique()
    tqdm.write("Split Complete (LOO):")
    tqdm.write(f"  Train Interactions: {len(train_df)}")
    tqdm.write(f"  Test Interactions:  {len(test_df)}")
    if n_users > 0:
        tqdm.write(f"  Avg Test Items/User: {len(test_df) / n_users:.2f} (Should be 1.0)")

    return train_df, test_df


def ratio_split(df, user_col='user_idx', test_ratio=0.2):
    """
    Splits 20% of each user's interactions into Test.
    """
    tqdm.write(f"Performing Ratio Split (Test Ratio: {test_ratio})...")

    # Randomize within user groups
    df = df.sample(frac=1, random_state=42).sort_values(user_col)

    # Assign rank to each item for each user
    df['rank'] = df.groupby(user_col).cumcount() + 1
    df['total'] = df.groupby(user_col)[user_col].transform('count')

    # Determine cutoff: Items > 80% of total go to Test
    cutoff = df['total'] * (1 - test_ratio)
    test_mask = df['rank'] > cutoff

    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    # Ensure overlap
    train_users = set(train_df[user_col].unique())
    test_df = test_df[test_df[user_col].isin(train_users)]

    tqdm.write("Split Complete (Ratio):")
    tqdm.write(f"  Train Interactions: {len(train_df)}")
    tqdm.write(f"  Test Interactions:  {len(test_df)}")

    return train_df, test_df


def save_adj_list(df, path):
    """
    Saves as: User Item1 Item2 ...
    """
    tqdm.write(f"Saving to {path}...")
    grouped = df.groupby('user_idx')['item_idx'].apply(list)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for user_id, items in grouped.items():
            items_str = " ".join(map(str, items))
            f.write(f"{user_id} {items_str}\n")


# ---------------------------------------------------------
# Datasets
# ---------------------------------------------------------

def process_ml1m(args):
    out_dir = DATA_DIR / "ml-1m"
    raw_dir = out_dir / "raw"
    zip_path = raw_dir / "ml-1m.zip"
    download_file(ML1M_URL, zip_path)
    extract_zip(zip_path, raw_dir)

    dat_file = raw_dir / "ml-1m" / "ratings.dat"
    tqdm.write("Loading ML-1M...")
    df = pd.read_csv(dat_file, sep='::', header=None, names=['user', 'item', 'rating', 'time'], engine='python')
    df = df[df['rating'] >= 4].copy()

    if args.core > 0:
        df = iterative_k_core(df, k=args.core)

    df = remap_ids(df, 'user', 'item', out_dir)

    if args.split == 'loo':
        train_df, test_df = robust_loo_split(df)
    else:
        train_df, test_df = ratio_split(df, test_ratio=0.2)

    save_adj_list(train_df, out_dir / "train.txt")
    save_adj_list(test_df, out_dir / "test.txt")


def process_gowalla(args):
    out_dir = DATA_DIR / "gowalla"
    raw_dir = out_dir / "raw"
    gz_path = raw_dir / "loc-gowalla_totalCheckins.txt.gz"
    txt_path = raw_dir / "gowalla.txt"
    download_file(GOWALLA_URL, gz_path)
    if not txt_path.exists():
        extract_gzip(gz_path, txt_path)

    tqdm.write("Loading Gowalla...")
    df = pd.read_csv(txt_path, sep='\t', header=None, names=['user', 'time', 'lat', 'lon', 'item'])
    df['time'] = pd.to_datetime(df['time']).astype('int64') // 10 ** 9

    if args.core > 0:
        df = iterative_k_core(df, k=args.core)

    df = remap_ids(df, 'user', 'item', out_dir)

    if args.split == 'loo':
        train_df, test_df = robust_loo_split(df)
    else:
        train_df, test_df = ratio_split(df, test_ratio=0.2)

    save_adj_list(train_df, out_dir / "train.txt")
    save_adj_list(test_df, out_dir / "test.txt")


def process_yelp(args):
    if not args.raw_path:
        raise ValueError("Yelp requires --raw-path to yelp_academic_dataset_review.json. "
                         "Yelp cannot be downloaded automatically due to License/EULA. "
                         "Download on https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset. "
        )

    out_dir = DATA_DIR / "yelp"
    tqdm.write("Loading Yelp (Chunked)...")
    chunks = []
    for chunk in tqdm(pd.read_json(args.raw_path, lines=True, chunksize=100000)):
        chunk = chunk[chunk['stars'] >= 4][['user_id', 'business_id', 'date']]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df.rename(columns={'user_id': 'user', 'business_id': 'item'}, inplace=True)
    df['time'] = pd.to_datetime(df['date']).astype('int64') // 10 ** 9

    # Use higher K-core (10 or 20)
    k = args.core if args.core > 0 else 10
    df = iterative_k_core(df, k=k)

    df = remap_ids(df, 'user', 'item', out_dir)

    if args.split == 'loo':
        train_df, test_df = robust_loo_split(df)
    else:
        train_df, test_df = ratio_split(df, test_ratio=0.2)

    save_adj_list(train_df, out_dir / "train.txt")
    save_adj_list(test_df, out_dir / "test.txt")


def process_amazon(args):
    out_dir = DATA_DIR / "amazon"
    raw_dir = out_dir / "raw"
    csv_path = raw_dir / "ratings_Books.csv"

    if not args.raw_path:
        if not csv_path.exists():
            tqdm.write("No --raw-path provided. Attempting to download Amazon Books dataset (3GB+)...")
            download_file(AMAZON_URL, csv_path)
        args.raw_path = str(csv_path)

    tqdm.write(f"Loading Amazon from {args.raw_path}...")

    # Amazon CSV: user, item, rating, timestamp
    df = pd.read_csv(args.raw_path, header=None, names=['user', 'item', 'rating', 'time'])

    tqdm.write("Filtering ratings >= 4...")
    df = df[df['rating'] >= 4].copy()

    # Amazon is extremely sparse, needs 10 or 20-core
    k = args.core if args.core > 0 else 10
    df = iterative_k_core(df, k=k)

    df = remap_ids(df, 'user', 'item', out_dir)

    if args.split == 'loo':
        train_df, test_df = robust_loo_split(df)
    else:
        train_df, test_df = ratio_split(df, test_ratio=0.2)

    save_adj_list(train_df, out_dir / "train.txt")
    save_adj_list(test_df, out_dir / "test.txt")


# ---------------------------------------------------------
# Main Entry
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=['ml-1m', 'gowalla', 'yelp', 'amazon'])
    parser.add_argument("--raw-path", help="Path to raw file for Yelp (or Amazon if you have it locally)")
    parser.add_argument("--core", type=int, default=10, help="K-core filter size")
    # Leave-One-Out (LOO) or ratio %20
    parser.add_argument("--split", choices=['loo', 'ratio'], default='loo', help="loo (1 item test) or ratio (20%% test)")
    args = parser.parse_args()

    if args.dataset == 'ml-1m':
        process_ml1m(args)
    elif args.dataset == 'gowalla':
        process_gowalla(args)
    elif args.dataset == 'yelp':
        process_yelp(args)
    elif args.dataset == 'amazon':
        process_amazon(args)