"""Run the paired hard-PCA update-schedule experiment for Figures 2 and 3.

The conditions use identical data, initial weights, optimizer settings, and
model architecture.  The only difference is how often the intermediate hard
PCA layer is recalibrated:

* ``no_schedule``: update at the first mini-batch of every epoch;
* ``scheduled``: update at the first mini-batch of selected epochs;
* ``both_fixed`` (optional): update only once, at epoch 1.

For ``PCA_NN_PCA`` and ``PCA_NN_PCA_ADD``, the first PCA layer uses
``initialize_once=True`` and the experiment isolates the second layer. The
``PCA_NN`` specification is an initialize-once control, whereas ``NN_PCA_NN``
places the scheduled hard-PCA layer inside the network.

The model and train/test loops are imported from the current replication code.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPLICATION_DIR = Path(__file__).resolve().parents[2]
if str(REPLICATION_DIR) not in sys.path:
    sys.path.insert(0, str(REPLICATION_DIR))

from data.covariate_standardized import FactorModel
from data.fast_data_standardized import HierarchicalCompositionModels
from methods.stat_methods import test_loop, train_loop
from models.model_lib_PCA import (
    PCA_NN,
    PCA_NN_PCA,
    PCA_NN_PCA_ADD,
    PcaLayer,
)


DEFAULT_SCHEDULE = [1, 3, 5, 7, 9, 11, 13, 15, 20, 30, 40]
MAIN_LR_RANGE = (1e-3, 1e-2)
MAIN_DEPTH_RANGE = (3, 4)


class NN_PCA_NN(nn.Module):
    """Hard-PCA architecture with the scheduled PCA layer inside the network."""

    def __init__(
        self,
        p: int,
        r_bar: int,
        depth: int,
        width: int,
        input_dropout: bool = False,
        dropout_rate: float = 0.0,
        check_depth: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if check_depth:
            assert depth >= 3
        self.use_input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.pre_nn_stack = nn.Sequential(
            OrderedDict(
                [
                    ("pre_linear1", nn.Linear(p, p)),
                    ("pre_relu1", nn.LeakyReLU(0.1)),
                ]
            )
        )
        self.pca_layer = PcaLayer(p, r_bar)
        layers = [
            ("linear2", nn.Linear(r_bar, width)),
            ("relu2", nn.LeakyReLU(0.1)),
        ]
        for layer_index in range(3, depth):
            layers.append((f"linear{layer_index}", nn.Linear(width, width)))
            layers.append((f"relu{layer_index}", nn.LeakyReLU(0.1)))
        layers.append((f"linear{depth}", nn.Linear(width, 1)))
        self.relu_stack = nn.Sequential(OrderedDict(layers))

    def forward(
        self, x: torch.Tensor, is_training: bool = False, initializing: bool = False, **kwargs
    ) -> torch.Tensor:
        if self.use_input_dropout and is_training:
            x = self.input_dropout(x)
        x = self.pre_nn_stack(x)
        x, _ = self.pca_layer(
            x,
            initializing=initializing,
            record_proj=kwargs.get("record_proj", False),
            use_proj_mean=kwargs.get("use_proj_mean", False),
        )
        return self.relu_stack(x)


def parse_int_list(value: str) -> list[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return [int(item) for item in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hard-PCA models under alternative update schedules."
    )
    parser.add_argument("--seeds", type=parse_int_list, default=[100])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--schedule", type=parse_int_list, default=DEFAULT_SCHEDULE)
    parser.add_argument(
        "--model",
        choices=["pca_nn", "nn_pca_nn", "pca_nn_pca", "pca_nn_pca_add"],
        default="pca_nn_pca_add",
    )
    parser.add_argument("--include-both-fixed", action="store_true")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=0,
        help="Tune one common config before comparing schedules; 0 disables tuning.",
    )
    parser.add_argument(
        "--tune-condition",
        choices=["no_schedule", "scheduled", "both_fixed"],
        default="scheduled",
        help="Reference policy used to select the common config by validation MSE.",
    )
    parser.add_argument(
        "--tune-patience",
        type=int,
        default=100,
        help="Validation early-stopping patience within each Optuna trial.",
    )
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--r-bar", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--add-width", type=int, default=10)
    parser.add_argument("--add-depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--b-f", type=float, default=1.0)
    parser.add_argument("--b-u", type=float, default=1.0)
    parser.add_argument("--factor-id", type=int, default=0)
    parser.add_argument("--hcm-id", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--valid-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPLICATION_DIR / "logs" / "Simulation" / "pca_schedule_current",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(y, dtype=torch.float32, device=device),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def generate_data(
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, DataLoader], list[int]]:
    # This follows the current exp_simulation.py DGP and split construction.
    seed_everything(seed)
    factor_model = FactorModel(
        p=args.p,
        r=args.r,
        b_f=args.b_f,
        b_u=args.b_u,
        func_idx=args.factor_id,
        func_l=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    )
    response_model = HierarchicalCompositionModels(
        args.hcm_id, idx_l=[], normalize=False
    )

    def sample(n: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
        observation, factor, _ = factor_model.sample(n=n, latent=True)
        y = response_model.sample(factor)
        if noise:
            y = y + np.random.normal(0, noise, (n, 1))
        return observation, y

    x_train, y_train = sample(args.train_size, args.noise)
    x_valid, y_valid = sample(args.valid_size, args.noise)
    x_test, y_test = sample(args.test_size, 0.0)

    loaders = {
        "train": make_loader(x_train, y_train, args.batch_size, device),
        "valid": make_loader(x_valid, y_valid, args.batch_size, device),
        "test": make_loader(x_test, y_test, args.batch_size, device),
    }
    return loaders, [int(value) for value in response_model.func_idx]


def define_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    common_kwargs = {
        "p": args.p,
        "r_bar": args.r_bar,
        "depth": args.depth,
        "width": args.width,
        "device": str(device),
        "check_depth": True,
    }
    if args.model == "pca_nn":
        model = PCA_NN(**common_kwargs).to(device)
    elif args.model == "nn_pca_nn":
        model = NN_PCA_NN(**common_kwargs).to(device)
    elif args.model == "pca_nn_pca":
        model = PCA_NN_PCA(**common_kwargs).to(device)
    else:
        model = PCA_NN_PCA_ADD(
            add_width=args.add_width,
            add_depth=args.add_depth,
            **common_kwargs,
        ).to(device)
    return model


def scheduled_pca_layer(args: argparse.Namespace, model: nn.Module) -> PcaLayer:
    if args.model in {"pca_nn", "nn_pca_nn"}:
        return model.pca_layer
    return model.pca_layer2


def condition_update_epochs(
    args: argparse.Namespace, condition: str
) -> set[int]:
    if condition == "no_schedule":
        return set(range(1, args.epochs + 1))
    if condition == "scheduled":
        return set(args.schedule)
    if condition == "both_fixed":
        return {1}
    raise ValueError(f"unknown condition: {condition}")


def tune_common_config(
    args: argparse.Namespace,
    seed: int,
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> tuple[dict[str, float | int], pd.DataFrame, dict[str, object]]:
    """Select one config, then reuse it unchanged across all schedule policies."""
    update_epochs = condition_update_epochs(args, args.tune_condition)

    def objective(trial: optuna.Trial) -> float:
        trial_args = copy.copy(args)
        trial_args.lr = trial.suggest_float("lr", *MAIN_LR_RANGE, log=True)
        trial_args.depth = trial.suggest_int("depth", *MAIN_DEPTH_RANGE)

        # Every trial starts from the same initialization. This makes the
        # validation comparison reflect hyperparameters rather than RNG drift.
        seed_everything(seed + 10_000)
        model = define_model(trial_args, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=trial_args.lr)
        loss_fn = nn.MSELoss()
        best_valid_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0

        for epoch in range(1, args.epochs + 1):
            train_loop(
                loaders["train"],
                model,
                loss_fn,
                optimizer,
                initializing=epoch in update_epochs,
                reg_lambda=0,
            )
            valid_loss, _, _ = test_loop(loaders["valid"], model, loss_fn)
            valid_loss = float(valid_loss)
            trial.report(valid_loss, step=epoch)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= args.tune_patience:
                break

        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("epochs_run", epoch)
        return best_valid_loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)
    trials = study.trials_dataframe()
    metadata = {
        "n_trials": args.n_trials,
        "sampler": "TPESampler",
        "sampler_seed": seed,
        "reference_condition": args.tune_condition,
        "objective": "minimum validation MSE over training epochs",
        "patience": args.tune_patience,
        "search_space": {
            "lr": {"low": MAIN_LR_RANGE[0], "high": MAIN_LR_RANGE[1], "log": True},
            "depth": {"low": MAIN_DEPTH_RANGE[0], "high": MAIN_DEPTH_RANGE[1]},
            "r_bar": args.r_bar,
            "width": args.width,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
        },
        "best_trial": study.best_trial.number,
        "best_validation_mse": study.best_value,
        "best_params": study.best_params,
    }
    return study.best_params, trials, metadata


def run_condition(
    args: argparse.Namespace,
    seed: int,
    condition: str,
    update_epochs: set[int],
    base_state: dict[str, torch.Tensor],
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    # Reset stochastic state so the paired runs differ only in update timing.
    seed_everything(seed + 20_000)
    model = define_model(args, device)
    model.load_state_dict(base_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    rows: list[dict[str, object]] = []
    best_valid_loss = float("inf")
    best_valid_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    start = time.time()

    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        pca_update = epoch in update_epochs
        pca_layer = scheduled_pca_layer(args, model)
        projection_before = pca_layer.place_holder.weight.detach().clone()
        train_loss, _ = train_loop(
            loaders["train"],
            model,
            loss_fn,
            optimizer,
            initializing=pca_update,
            reg_lambda=0,
        )
        projection_after = pca_layer.place_holder.weight.detach()
        projection_change_l2 = torch.linalg.vector_norm(
            projection_after - projection_before
        ).item()
        valid_loss, _, _ = test_loop(loaders["valid"], model, loss_fn)
        if valid_loss < best_valid_loss:
            best_valid_loss = float(valid_loss)
            best_valid_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        rows.append(
            {
                "seed": seed,
                "condition": condition,
                "epoch": epoch,
                "pca_update": pca_update,
                "projection_change_l2": projection_change_l2,
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
            }
        )
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"seed={seed} condition={condition} epoch={epoch}/{args.epochs} "
                f"update={pca_update} projection_change={projection_change_l2:.6f} "
                f"train={train_loss:.6f} valid={valid_loss:.6f}",
                flush=True,
            )

    final_test_loss, _, _ = test_loop(loaders["test"], model, loss_fn)
    if best_state is None:
        raise RuntimeError("no best-validation checkpoint was recorded")
    model.load_state_dict(best_state)
    best_valid_test_loss, _, _ = test_loop(loaders["test"], model, loss_fn)
    summary = {
        "seed": seed,
        "condition": condition,
        "lr": args.lr,
        "depth": args.depth,
        "r_bar": args.r_bar,
        "width": args.width,
        "batch_size": args.batch_size,
        "best_valid_loss": best_valid_loss,
        "best_valid_epoch": best_valid_epoch,
        "final_train_loss": rows[-1]["train_loss"],
        "final_valid_loss": rows[-1]["valid_loss"],
        "test_loss_at_best_valid": float(best_valid_test_loss),
        "test_loss_at_final_epoch": float(final_test_loss),
        "elapsed_seconds": time.time() - start,
    }
    return rows, summary


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPLICATION_DIR.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if args.n_trials < 0:
        raise ValueError("--n-trials must be nonnegative")
    if args.tune_patience < 1:
        raise ValueError("--tune-patience must be positive")
    if min(args.schedule) < 1 or max(args.schedule) > args.epochs:
        raise ValueError("schedule epochs must lie in [1, epochs]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    response_functions: dict[str, list[int]] = {}
    tuning_by_seed: dict[str, dict[str, object]] = {}

    print(f"Using {device}; writing outputs to {output_dir}", flush=True)
    for seed in args.seeds:
        loaders, func_idx = generate_data(args, seed, device)
        response_functions[str(seed)] = func_idx

        if args.n_trials > 0:
            best_params, tuning_trials, tuning_metadata = tune_common_config(
                args, seed, loaders, device
            )
            args.lr = float(best_params["lr"])
            args.depth = int(best_params["depth"])
            tuning_by_seed[str(seed)] = tuning_metadata
            tuning_trials.insert(0, "seed", seed)
            tuning_trials.to_csv(
                output_dir / f"pca_schedule_tuning_trials_seed{seed}.csv",
                index=False,
            )
            print(
                f"seed={seed} selected common config {best_params} "
                f"using {args.tune_condition}",
                flush=True,
            )

        seed_everything(seed + 10_000)
        base_model = define_model(args, device)
        base_state = copy.deepcopy(base_model.state_dict())
        del base_model

        conditions = {
            "no_schedule": condition_update_epochs(args, "no_schedule"),
            "scheduled": condition_update_epochs(args, "scheduled"),
        }
        if args.include_both_fixed:
            conditions["both_fixed"] = condition_update_epochs(args, "both_fixed")
        for condition, update_epochs in conditions.items():
            rows, summary = run_condition(
                args,
                seed,
                condition,
                update_epochs,
                base_state,
                loaders,
                device,
            )
            all_rows.extend(rows)
            summaries.append(summary)

    history_path = output_dir / "pca_schedule_loss_history.csv"
    summary_path = output_dir / "pca_schedule_summary.csv"
    pd.DataFrame(all_rows).to_csv(history_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    config["schedule"] = list(args.schedule)
    config["seeds"] = list(args.seeds)
    config["device"] = str(device)
    config["git_commit"] = git_commit()
    model_names = {
        "pca_nn": "PCA_NN",
        "nn_pca_nn": "NN_PCA_NN",
        "pca_nn_pca": "PCA_NN_PCA",
        "pca_nn_pca_add": "PCA_NN_PCA_ADD",
    }
    model_name = model_names[args.model]
    if args.model == "nn_pca_nn":
        config["model_class"] = (
            "experiment.simulation.run_simulation_pca_schedule.NN_PCA_NN"
        )
    else:
        config["model_class"] = f"models.model_lib_PCA.{model_name}"
    config["response_function_indices"] = response_functions
    config["tuning_by_seed"] = tuning_by_seed
    config["epoch_numbering"] = "one-based"
    config["no_schedule_definition"] = "PCA update at first mini-batch of every epoch"
    config["scheduled_definition"] = "PCA update at first mini-batch of listed epochs"
    config["both_fixed_definition"] = "Both PCA layers update only at epoch 1"
    config["figure_specific_scheduled_pca_operation"] = True
    config["shared_model_patch_required"] = False
    config["initialize_once_control"] = args.model == "pca_nn"
    with (output_dir / "pca_schedule_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"Saved {history_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
