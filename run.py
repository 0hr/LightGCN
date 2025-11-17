import argparse
import copy

from models.lightgcn import run_lightgcn
from models.mfbpr import run_mf
from models.multivae import run_multvae
from models.ngcf import run_ngcf
from models.wmf import run_wmf
from utils import  *


# ----------------------------
# Runner
# ----------------------------

def run_models(args):
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    seeds = args.seeds if args.seeds else [args.seed]
    model_order = ["lightgcn", "ngcf", "mf-bpr", "wmf", "mult-vae"]
    if args.model is None or args.model == "all":
        models_to_run = model_order
    else:
        models_to_run = [args.model]

    final_results = {}

    for model_name in models_to_run:
        all_metrics = []
        model_plots = []
        seed_times = []
        for seed in seeds:
            if model_name == "lightgcn":
                metrics, history, train_time = run_lightgcn(args, seed, device)
            elif model_name == "ngcf":
                metrics, history, train_time = run_ngcf(args, seed, device)
            elif model_name == "mf-bpr":
                metrics, history, train_time = run_mf(args, seed, device)
            elif model_name == "wmf":
                metrics, history, train_time = run_wmf(args, seed, device)
            elif model_name == "mult-vae":
                metrics, history, train_time = run_multvae(args, seed, device)
            else:
                raise SystemExit(f"Unknown model {model_name}")
            all_metrics.append(metrics)
            seed_times.append(train_time)
            if history:
                plot_file = plot_metric_history(history, model_name, args.dataset, args.plot_dir)
                if plot_file:
                    model_plots.append(plot_file)
                time_plot = plot_time_history(history, model_name, args.dataset, args.plot_dir)
                if time_plot:
                    model_plots.append(time_plot)
                loss_plot = plot_loss_history(history, model_name, args.dataset, args.plot_dir)
                if loss_plot:
                    model_plots.append(loss_plot)

            if torch.cuda.is_available() and not args.cpu:
                max_mem = torch.cuda.max_memory_allocated(device)
                tqdm.write(f"[{model_name}][Seed {seed}] peak CUDA memory: {max_mem / 1e6:.1f} MB")
                torch.cuda.reset_peak_memory_stats(device)

        if all_metrics:
            if len(all_metrics) > 1:
                avg = {k: float(np.mean([m.get(k, 0.0) for m in all_metrics])) for k in all_metrics[0].keys()}
                avg["train_time_s"] = float(np.mean(seed_times))
                tqdm.write(f"[{model_name}] Averaged over seeds {seeds}: {avg}")
                final_results[model_name] = avg
            else:
                single = dict(all_metrics[0])
                single["train_time_s"] = float(seed_times[0]) if seed_times else 0.0
                tqdm.write(f"[{model_name}] Metrics: {single}")
                final_results[model_name] = single

    if final_results:
        tqdm.write("\n== Combined results ==")
        tqdm.write(format_metrics_table(final_results))
        if args.plot_dir:
            times = {m: vals.get("train_time_s", 0.0) for m, vals in final_results.items()}
            plot_time_bar(times, Path(args.plot_dir) / f"{args.dataset}_train_times.png")
    return final_results


def run_cv(args):
    if args.cv_folds < 2:
        return run_models(args)

    data_root = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent / "data" / args.dataset
    cv_seed = args.cv_seed if args.cv_seed is not None else args.seed
    folds = build_cv_splits(data_root, args.cv_folds, cv_seed)

    tqdm.write(f"Running {args.cv_folds}-fold cross-validation (seed {cv_seed}) on {args.dataset}")
    all_fold_results = []
    agg_results = defaultdict(list)

    for fold_idx, (train_ui, test_ui) in enumerate(folds, start=1):
        fold_dir = Path(args.plot_dir) / f"cv_fold_{fold_idx}"
        fold_data_dir = fold_dir / "data"
        write_user_items(train_ui, fold_data_dir / "train.txt")
        write_user_items(test_ui, fold_data_dir / "test.txt")

        args_fold = copy.deepcopy(args)
        args_fold.data_dir = str(fold_data_dir)
        args_fold.plot_dir = fold_dir

        tqdm.write(f"\n=== Fold {fold_idx}/{args.cv_folds} ===")
        fold_results = run_models(args_fold)
        all_fold_results.append(fold_results)
        for model_name, metrics in fold_results.items():
            agg_results[model_name].append(metrics)

    if agg_results:
        averaged = {}
        for model_name, entries in agg_results.items():
            keys = set()
            for m in entries:
                keys.update(m.keys())
            avg = {k: float(np.mean([m.get(k, 0.0) for m in entries])) for k in keys}
            averaged[model_name] = avg

        tqdm.write("\n== Cross-validated averages ==")
        tqdm.write(format_metrics_table(averaged))
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train LightGCN and baselines on implicit data")
    parser.add_argument("--dataset", required=True, choices=["amazon", "gowalla", "yelp", "ml-1m"], help="Dataset name")
    parser.add_argument(
        "--model",
        choices=["lightgcn",  "ngcf", "mf-bpr", "wmf", "mult-vae", "all"],
        help="Model to train; omit or set to 'all' to run every model",
    )
    parser.add_argument("--plot-dir", type=Path, default=Path("plots"), help="Where to save metric plots")
    parser.add_argument("--data-dir", help="Path to dataset folder (default: ./data/<dataset>)")
    parser.add_argument("--max-users", type=int, help="Limit users loaded from the dataset for quick runs")
    parser.add_argument("--max-items-per-user", type=int, help="Cap training interactions kept per user when subsetting")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension / latent size")
    parser.add_argument("--layers", type=int, default=3, help="GCN/NGCF propagation layers")
    parser.add_argument("--learnable-weights", action="store_true", help="Use learnable layer weights (LightGCN)", default=False)
    parser.add_argument("--edge-dropout", type=float, default=0.0, help="Edge dropout rate for robustness")
    parser.add_argument("--neg-sampler", choices=["uniform", "popularity"], default="uniform", help="Negative sampling strategy")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--steps-per-epoch", type=int, default=1024, help="Max steps per epoch to limit compute")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument("--use_batch_l2", type=bool, default=False, help="Use L2 weight decay in lightgcn")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience on NDCG@20")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--seed", type=int, default=42, help="Primary seed")
    parser.add_argument("--seeds", nargs="*", type=int, help="Optional list of seeds to average over")

    # WMF/Mult-VAE specifics
    parser.add_argument("--wmf-alpha", type=float, default=1.0, help="Confidence scaling for WMF")
    parser.add_argument("--vae-hidden", type=int, default=600, help="VAE hidden layer size")
    parser.add_argument("--vae-beta", type=float, default=0.2, help="Beta for VAE KL term")
    parser.add_argument("--vae-dropout", type=float, default=0.5, help="Dropout rate for VAE encoder")
    parser.add_argument("--cv-folds", type=int, default=1, help="Number of user-wise CV folds (>=2 to enable)")
    parser.add_argument("--cv-seed", type=int, help="Seed used for CV shuffling (default: --seed)")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_cv(args)
