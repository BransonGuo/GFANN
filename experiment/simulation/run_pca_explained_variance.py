"""Reproduce the explained-variance diagnostic for paper Figure 4.

The experiment reuses a configuration selected under the scheduled hard-PCA
specification. During the paired schedule experiment, the first PCA layer of
``PCA_NN_PCA`` is initialized once and only the second PCA layer follows the
operating schedule. This script reruns the scheduled condition and records two
full-training-sample diagnostics after every epoch:

* variance captured by the second PCA layer's current basis; and
* variance captured by an oracle top-k PCA basis recomputed for diagnostics.

The diagnostic eigendecompositions are performed under ``torch.no_grad()`` and
never modify the model or its PCA schedule.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from run_simulation_pca_schedule import (
    define_model,
    generate_data,
    parse_int_list,
    scheduled_pca_layer,
    seed_everything,
    test_loop,
    train_loop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the PCA explained-variance diagnostic."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        type=Path,
        help="pca_schedule_config.json from the scheduled Optuna experiment.",
    )
    source.add_argument(
        "--history",
        type=Path,
        help=(
            "Saved figure4_explained_variance_history.csv. This redraws "
            "Figure 4 without retraining the model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. With --config, defaults to a "
            "figure4_explained_variance directory beside the config; with "
            "--history, defaults to the history file's directory."
        ),
    )
    parser.add_argument(
        "--retained-components",
        type=int,
        default=10,
        help="Number of retained PCA components when plotting saved history.",
    )
    parser.add_argument(
        "--schedule",
        type=parse_int_list,
        default=None,
        help=(
            "Optional comma-separated PCA update epochs. By default, use the "
            "schedule stored in the source configuration."
        ),
    )
    return parser.parse_args()


def namespace_from_config(config: dict[str, object]) -> argparse.Namespace:
    required = {
        "seeds",
        "epochs",
        "schedule",
        "model",
        "p",
        "r",
        "r_bar",
        "depth",
        "width",
        "add_width",
        "add_depth",
        "batch_size",
        "lr",
        "noise",
        "b_f",
        "b_u",
        "factor_id",
        "hcm_id",
        "train_size",
        "valid_size",
        "test_size",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"configuration is missing keys: {sorted(missing)}")
    if config["model"] != "pca_nn_pca":
        raise ValueError("Figure 4 currently requires model='pca_nn_pca'")
    seeds = list(config["seeds"])
    if len(seeds) != 1:
        raise ValueError("Figure 4 is an illustrative single-seed diagnostic")
    return argparse.Namespace(**config)


def hidden_before_second_pca(
    model: nn.Module,
    x_train: torch.Tensor,
) -> torch.Tensor:
    """Return the full-sample representation entering the second PCA layer."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        first_scores, _ = model.pca_layer(x_train, initializing=False)
        hidden = model.relu_stack(first_scores)
    model.train(was_training)
    return hidden


def explained_variance_metrics(
    model: nn.Module,
    x_train: torch.Tensor,
    k: int,
) -> dict[str, float]:
    """Measure optimal and current-basis explained variance on fixed data."""
    hidden = hidden_before_second_pca(model, x_train).to(torch.float64)
    hidden = hidden - hidden.mean(dim=0, keepdim=True)
    covariance = hidden.T @ hidden / max(hidden.shape[0] - 1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total_variance = eigenvalues.sum()
    if not torch.isfinite(total_variance) or total_variance <= 0:
        raise RuntimeError("hidden representation has non-positive total variance")

    k_eff = min(k, eigenvalues.numel())
    optimal_topk_evr = eigenvalues[-k_eff:].sum() / total_variance

    pca_layer = model.pca_layer2
    current_basis = getattr(pca_layer.PcaOperation, "P", None)
    if current_basis is None:
        current_basis = pca_layer.place_holder.weight.detach().T
    current_basis = current_basis.to(dtype=torch.float64)
    orthonormal_basis = torch.linalg.qr(current_basis, mode="reduced").Q
    captured_variance = torch.trace(
        orthonormal_basis.T @ covariance @ orthonormal_basis
    )
    current_basis_evr = captured_variance / total_variance

    return {
        "current_basis_evr": float(current_basis_evr.clamp(0.0, 1.0)),
        "optimal_topk_evr": float(optimal_topk_evr.clamp(0.0, 1.0)),
        "evr_gap": float((optimal_topk_evr - current_basis_evr).clamp_min(0.0)),
        "hidden_total_variance": float(total_variance),
    }


def plot_diagnostic(
    history: pd.DataFrame,
    output_path: Path,
    include_optimal: bool,
    retained_components: int,
) -> None:
    fig, loss_axis = plt.subplots(figsize=(7.2, 5.2))
    variance_axis = loss_axis.twinx()

    loss_axis.plot(
        history["epoch"],
        history["train_loss"],
        color="#1f77b4",
        linewidth=1.7,
        label="Training error",
    )
    loss_axis.plot(
        history["epoch"],
        history["valid_loss"],
        color="#ff7f0e",
        linewidth=1.7,
        label="Validation error",
    )
    variance_axis.plot(
        history["epoch"],
        100.0 * history["current_basis_evr"],
        color="#2ca02c",
        linewidth=1.8,
        label=f"Current {retained_components}-dimensional basis EVR",
    )
    if include_optimal:
        variance_axis.plot(
            history["epoch"],
            100.0 * history["optimal_topk_evr"],
            color="#006d2c",
            linestyle="--",
            linewidth=1.4,
            label=f"Oracle top-{retained_components} EVR",
        )

    for epoch in history.loc[history["pca_update"], "epoch"]:
        loss_axis.axvline(
            epoch,
            color="#7f8c8d",
            alpha=0.12,
            linewidth=0.8,
            zorder=0,
        )

    minimum_evr = 100.0 * history[
        ["current_basis_evr", "optimal_topk_evr"]
    ].min().min()
    lower_limit = max(0.0, 5.0 * math.floor((minimum_evr - 3.0) / 5.0))
    variance_axis.set_ylim(lower_limit, 100.0)
    loss_axis.set_xlim(1, int(history["epoch"].max()))
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Mean squared error")
    variance_axis.set_ylabel("Explained variance ratio (%)", color="#238b45")
    variance_axis.tick_params(axis="y", labelcolor="#238b45")
    loss_axis.grid(alpha=0.18, linewidth=0.6)

    handles_left, labels_left = loss_axis.get_legend_handles_labels()
    handles_right, labels_right = variance_axis.get_legend_handles_labels()
    fig.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        bbox_to_anchor=(0.5, 0.995),
        loc="upper center",
        ncol=2,
        fontsize=9,
        frameon=True,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cli_args = parse_args()
    if cli_args.history is not None:
        history_path = cli_args.history.resolve()
        output_dir = (
            cli_args.output_dir.resolve()
            if cli_args.output_dir
            else history_path.parent
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        history = pd.read_csv(history_path)
        required = {
            "epoch",
            "pca_update",
            "train_loss",
            "valid_loss",
            "current_basis_evr",
            "optimal_topk_evr",
        }
        missing = required.difference(history.columns)
        if missing:
            raise ValueError(
                f"saved Figure 4 history is missing columns: {sorted(missing)}"
            )
        plot_diagnostic(
            history,
            output_dir / "pct_variance_explained_with_train_valid.PNG",
            include_optimal=True,
            retained_components=cli_args.retained_components,
        )
        print(f"Redrew Figure 4 from saved history in {output_dir}", flush=True)
        return

    assert cli_args.config is not None
    config_path = cli_args.config.resolve()
    with config_path.open(encoding="utf-8") as file:
        source_config = json.load(file)
    args = namespace_from_config(source_config)
    if cli_args.schedule is not None:
        args.schedule = cli_args.schedule
    if min(args.schedule) < 1 or max(args.schedule) > int(args.epochs):
        raise ValueError("schedule epochs must lie in [1, epochs]")
    output_dir = (
        cli_args.output_dir.resolve()
        if cli_args.output_dir
        else config_path.parent / "figure4_explained_variance"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(args.seeds[0])
    loaders, response_functions = generate_data(args, seed, device)
    x_train = loaders["train"].dataset.tensors[0]

    seed_everything(seed + 10_000)
    base_model = define_model(args, device)
    base_state = copy.deepcopy(base_model.state_dict())
    del base_model

    seed_everything(seed + 20_000)
    model = define_model(args, device)
    model.load_state_dict(base_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()
    update_epochs = set(int(epoch) for epoch in args.schedule)
    rows: list[dict[str, object]] = []
    best_valid_loss = float("inf")
    best_valid_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, int(args.epochs) + 1):
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
        projection_change = torch.linalg.vector_norm(
            pca_layer.place_holder.weight.detach() - projection_before
        ).item()
        valid_loss, _, _ = test_loop(loaders["valid"], model, loss_fn)
        metrics = explained_variance_metrics(model, x_train, int(args.r_bar))
        if valid_loss < best_valid_loss:
            best_valid_loss = float(valid_loss)
            best_valid_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "pca_update": pca_update,
                "projection_change_l2": projection_change,
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                **metrics,
            }
        )
        if epoch == 1 or epoch % 25 == 0 or epoch == int(args.epochs):
            print(
                f"epoch={epoch}/{args.epochs} update={pca_update} "
                f"train={train_loss:.6f} valid={float(valid_loss):.6f} "
                f"current_evr={metrics['current_basis_evr']:.4f} "
                f"optimal_evr={metrics['optimal_topk_evr']:.4f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("no best-validation checkpoint was recorded")
    final_test_loss, _, _ = test_loop(loaders["test"], model, loss_fn)
    model.load_state_dict(best_state)
    best_valid_test_loss, _, _ = test_loop(loaders["test"], model, loss_fn)

    history = pd.DataFrame(rows)
    history_path = output_dir / "figure4_explained_variance_history.csv"
    history.to_csv(history_path, index=False)
    plot_diagnostic(
        history,
        output_dir / "pct_variance_explained_with_train_valid.PNG",
        include_optimal=True,
        retained_components=int(args.r_bar),
    )
    plot_diagnostic(
        history,
        output_dir / "pca_explained_variance_audit.PNG",
        include_optimal=True,
        retained_components=int(args.r_bar),
    )

    summary = {
        "source_config": config_path.name,
        "model": args.model,
        "dataset": "DS21" if int(args.factor_id) == 1 else "DS11",
        "seed": seed,
        "p": int(args.p),
        "r_bar": int(args.r_bar),
        "epochs": int(args.epochs),
        "schedule": sorted(update_epochs),
        "lr": float(args.lr),
        "depth": int(args.depth),
        "width": int(args.width),
        "batch_size": int(args.batch_size),
        "response_function_indices": response_functions,
        "best_valid_loss": best_valid_loss,
        "best_valid_epoch": best_valid_epoch,
        "test_loss_at_best_valid": float(best_valid_test_loss),
        "test_loss_at_final_epoch": float(final_test_loss),
        "evr_sample": "complete fixed training sample",
        "current_basis_evr_definition": "trace(Q.T @ covariance @ Q) / trace(covariance)",
        "optimal_topk_evr_definition": "sum(largest k eigenvalues) / trace(covariance)",
    }
    with (output_dir / "figure4_explained_variance_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"Saved Figure 4 diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
