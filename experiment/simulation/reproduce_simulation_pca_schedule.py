"""Plot the Figure 2/3 PCA-schedule diagnostics from saved loss histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "Simulation"
    / "pca_schedule_current"
    / "pca_schedule_loss_history.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the hard-PCA schedule training diagnostics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def aggregate_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {
        "seed",
        "condition",
        "epoch",
        "pca_update",
        "projection_change_l2",
        "train_loss",
        "valid_loss",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    long = history.melt(
        id_vars=["seed", "condition", "epoch", "pca_update"],
        value_vars=["train_loss", "valid_loss"],
        var_name="series",
        value_name="loss",
    )
    grouped = long.groupby(["condition", "epoch", "series"], as_index=False)
    aggregate = grouped["loss"].agg(["mean", "std", "count"]).reset_index()
    aggregate["sem"] = aggregate["std"].fillna(0.0) / np.sqrt(aggregate["count"])
    return aggregate


def plot_condition(
    aggregate: pd.DataFrame,
    history: pd.DataFrame,
    condition: str,
    output_path: Path,
    y_limits: tuple[float, float],
) -> None:
    labels = {
        "train_loss": "Training error",
        "valid_loss": "Validation error",
    }
    colors = {"train_loss": "#1f77b4", "valid_loss": "#ff7f0e"}
    fig, axis = plt.subplots(figsize=(7.2, 5.2))

    for series in ["train_loss", "valid_loss"]:
        data = aggregate[
            (aggregate["condition"] == condition) & (aggregate["series"] == series)
        ].sort_values("epoch")
        x = data["epoch"].to_numpy()
        mean = data["mean"].to_numpy()
        sem = data["sem"].to_numpy()
        axis.plot(x, mean, color=colors[series], linewidth=1.7, label=labels[series])
        if int(data["count"].max()) > 1:
            axis.fill_between(
                x,
                mean - 1.96 * sem,
                mean + 1.96 * sem,
                color=colors[series],
                alpha=0.16,
                linewidth=0,
            )

    if condition == "scheduled":
        update_epochs = sorted(
            history.loc[
                (history["condition"] == condition) & history["pca_update"], "epoch"
            ].unique()
        )
        for epoch in update_epochs:
            axis.axvline(epoch, color="#7f8c8d", alpha=0.12, linewidth=0.8, zorder=0)

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_xlim(1, int(history["epoch"].max()))
    axis.set_ylim(*y_limits)
    axis.legend(frameon=True)
    axis.grid(alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = (args.output_dir or input_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(input_path)
    aggregate = aggregate_history(history)
    aggregate.to_csv(output_dir / "pca_schedule_loss_history_seed_average.csv", index=False)

    ymax = float(history[["train_loss", "valid_loss"]].max().max())
    y_limits = (0.0, ymax * 1.04 if ymax > 0 else 1.0)
    plot_condition(
        aggregate,
        history,
        "no_schedule",
        output_dir / "train_valid_loss_no_schedule.PNG",
        y_limits,
    )
    plot_condition(
        aggregate,
        history,
        "scheduled",
        output_dir / "train_valid_loss_schedule.PNG",
        y_limits,
    )
    print(f"Saved figures and aggregate data to {output_dir}")


if __name__ == "__main__":
    main()
