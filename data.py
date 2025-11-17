#!/usr/bin/env python3

import argparse
import contextlib
import shutil
import sys
from pathlib import Path
from urllib import error, request
from tqdm.auto import tqdm

DEFAULT_BASE_URL = "https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/refs/heads/master/data"

DATASETS = {
    "amazon": {
        "folder": "amazon",
        "remote_dir": "amazon-book",
        "files": ("train.txt", "test.txt"),
        "optional_files": ("user_list.txt", "item_list.txt"),
    },
    "gowalla": {
        "folder": "gowalla",
        "remote_dir": "gowalla",
        "files": ("train.txt", "test.txt"),
        "optional_files": ("user_list.txt", "item_list.txt"),
    },
    "yelp": {
        "folder": "yelp",
        "remote_dir": "yelp2018",
        "files": ("train.txt", "test.txt"),
        "optional_files": ("user_list.txt", "item_list.txt"),
    },
}


class DownloadError(RuntimeError):
    pass

def _download_file(url, destination, overwrite):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        tqdm.write(f"{destination} exists, skipping (use --force to re-download)")
        return

    tqdm.write(f"{url} -> {destination}")
    with contextlib.closing(request.urlopen(url)) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    tqdm.write(f"Saved {destination}")


def download_dataset(name, base_url, overwrite=False):
    name = name.lower()
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Valid options: {', '.join(DATASETS)}")

    cfg = DATASETS[name]
    remote_dir = cfg.get("remote_dir", name)
    data_root = Path(__file__).resolve().parent / "data" / cfg.get("folder", name)
    required_files = list(cfg.get("files", ()))
    optional_files = list(cfg.get("optional_files", ()))

    for filename in required_files + optional_files:
        url = f"{base_url.rstrip('/')}/{remote_dir}/{filename}"
        destination = data_root / filename
        try:
            _download_file(url, destination, overwrite)
        except error.HTTPError as exc:
            if filename in optional_files and exc.code == 404:
                tqdm.write(f"Missing optional file {filename} ({exc.code}), continuing...")
                continue
            raise DownloadError(f"Failed to fetch {filename} for {name}: {exc}") from exc

    return data_root


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Download LightGCN sample datasets (amazon, gowalla, movielens, yelp)."
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dataset", choices=DATASETS.keys(), help="Dataset to download")
    for dataset in DATASETS:
        group.add_argument(f"--{dataset}", action="store_true", help=f"Shortcut for --dataset {dataset}")

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL hosting dataset folders (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist")
    parser.add_argument("--list", action="store_true", help="List supported datasets and exit")
    return parser.parse_args(argv)


def _selected_dataset(args):
    if args.dataset:
        return args.dataset
    for dataset in DATASETS:
        if getattr(args, dataset, False):
            return dataset
    return None


def main(argv=None):
    args = _parse_args(argv or sys.argv[1:])
    if args.list:
        tqdm.write("Available datasets:")
        for name, cfg in DATASETS.items():
            folder = cfg.get("folder", name)
            remote_dir = cfg.get("remote_dir", folder)
            tqdm.write(f"- {name}: downloads to ./data/{folder} (remote folder: {remote_dir})")
        return 0

    dataset = _selected_dataset(args)
    if dataset is None:
        tqdm.write("Please pick a dataset using --dataset, --amazon, --gowalla, --movielens or --yelp.", file=sys.stderr)
        return 1

    try:
        target_dir = download_dataset(dataset, args.base_url, overwrite=args.force)
    except DownloadError as exc:
        tqdm.write(exc, file=sys.stderr)
        return 2

    tqdm.write(f"Finished downloading {dataset} into {target_dir}")
    return 0


if __name__ == "__main__":
    result = main()
    sys.exit(result)
