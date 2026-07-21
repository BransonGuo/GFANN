"""Run and save the per-seed Optuna studies used for paper Figure 9."""

import argparse
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EXPERIMENT_FILE = SCRIPT_DIR / "exp_simulation_hyperparameter_importance.py"
PAPER_SEEDS = tuple(range(100, 150, 5))
MODELS = (
    "fAR-NNOpt",
    "pCA_NN_PCA_ADDOpt",
    "sPCA_NN_SPCA_ADDOpt",
)


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("expected a comma-separated seed list")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 10-seed Optuna studies used for Figure 9."
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=PAPER_SEEDS,
        help="Comma-separated seeds; defaults to 100,105,...,145.",
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional directory name under logs/Simulation/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and output paths without starting training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S.%f")
    run_name = args.run_name or (
        f"{timestamp}Simulation_hyperparameter_importance_trial{args.n_trials}"
    )
    log_root = Path("Simulation") / run_name
    output_root = REPO_ROOT / "logs" / log_root
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for seed in args.seeds:
        seed_dir = f"figure9_seed{seed}_trial{args.n_trials}"
        common_args = [
            sys.executable,
            str(EXPERIMENT_FILE),
            "--suffix",
            "figure9",
            "--p",
            "1000",
            "--n",
            "500",
            "--num_epoch",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--min_delta",
            str(args.min_delta),
            "--n_trials",
            str(args.n_trials),
            "--y_func_l",
            "[0,1,2,3,4,5,6,7]",
            "--factor_id",
            "0",
            "--hcm_id",
            "2",
            "--seed",
            str(seed),
            "--save-study",
            "--sensitivity-search-space",
        ]
        print(f"Starting Figure 9 seed {seed}", flush=True)
        for model in MODELS:
            model_log_dir = log_root / seed_dir / model
            model_output_dir = REPO_ROOT / "logs" / model_log_dir
            if any(model_output_dir.glob("*_study_*.pkl")):
                print(f"Skipping completed seed {seed}, model {model}", flush=True)
                continue
            command = common_args + [
                "--log_dir",
                str(model_log_dir),
                "--model_l",
                model,
            ]
            if args.dry_run:
                print(shlex.join(command), flush=True)
            else:
                subprocess.run(command, cwd=REPO_ROOT, check=True)

    elapsed_hours = (time.time() - start) / 3600
    status = "Prepared" if args.dry_run else "Completed"
    print(f"{status} Figure 9 studies in {elapsed_hours:.2f} hours", flush=True)
    print(f"Run root: {output_root}", flush=True)
    print(
        "Reproduce with: "
        f"{sys.executable} {SCRIPT_DIR / 'reproduce_simulation_hyperparameter_importance.py'} "
        f"--run-root {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
