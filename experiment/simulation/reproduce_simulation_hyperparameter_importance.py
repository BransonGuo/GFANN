from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.trial import TrialState


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figure9_outputs"

MODEL_SPECS = {
    "fAR-NNOpt": {
        "label": "FAR-NN",
        "color": "#1f77b4",
        "filename": "hyperpara_farnn.png",
    },
    "pCA_NN_PCA_ADDOpt": {
        "label": "PCA_NN_PCA_ADD",
        "color": "#ff7f0e",
        "filename": "hyperpara_ori_pca_pca_add.png",
    },
    "sPCA_NN_SPCA_ADDOpt": {
        "label": "SPCA_NN_SPCA_ADD",
        "color": "#d62728",
        "filename": "hyperpara_pca_pca_add.png",
    },
}

PARAMETER_LABELS = {
    "lr": "Learning rate",
    "width": "Main width",
    "r_bar": r"Working dimension ($\bar r$)",
    "depth": "Depth",
    "add_width": "Additive width",
    "add_depth": "Additive depth",
    "lambda_pca": r"$\lambda_{\mathrm{PCA}}$",
    "lambda_orthogonality": r"$\lambda_{\mathrm{orth}}$",
}


def resolve_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def seed_from_run_dir(run_dir: Path) -> int:
    match = re.search(r"_seed(\d+)(?:_|$)", run_dir.name)
    if match is None:
        raise ValueError(f"Cannot infer seed from run-directory name: {run_dir.name}")
    return int(match.group(1))


def varying_parameters(study: optuna.Study) -> set[str]:
    values: dict[str, set[str]] = {}
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue
        for parameter, value in trial.params.items():
            values.setdefault(parameter, set()).add(repr(value))
    return {parameter for parameter, observed in values.items() if len(observed) > 1}


def read_importances(
    run_dirs: list[Path], expected_trials: int, fanova_seed: int
) -> pd.DataFrame:
    records: list[dict] = []
    seen_seeds: set[int] = set()

    for run_dir in run_dirs:
        seed = seed_from_run_dir(run_dir)
        if seed in seen_seeds:
            raise ValueError(f"Seed {seed} appears in more than one run directory")
        seen_seeds.add(seed)

        for model in MODEL_SPECS:
            studies = list((run_dir / model).glob("*_study_*.pkl"))
            if len(studies) != 1:
                raise ValueError(
                    f"Expected one study for {model} under {run_dir}; found {len(studies)}"
                )
            study_path = studies[0]
            study = joblib.load(study_path)
            complete_trials = sum(
                trial.state == TrialState.COMPLETE for trial in study.trials
            )
            if complete_trials != expected_trials:
                raise ValueError(
                    f"{study_path} has {complete_trials} completed trials; "
                    f"expected {expected_trials}"
                )

            importance = get_param_importances(
                study,
                evaluator=FanovaImportanceEvaluator(seed=fanova_seed),
            )
            variable = varying_parameters(study)
            for parameter, value in importance.items():
                if parameter not in variable:
                    continue
                records.append(
                    {
                        "seed": seed,
                        "model": model,
                        "model_label": MODEL_SPECS[model]["label"],
                        "parameter": parameter,
                        "importance": value,
                        "complete_trials": complete_trials,
                        "study_file": str(Path(run_dir.name) / model / study_path.name),
                    }
                )

    if not records:
        raise ValueError("No hyperparameter importances were recovered")
    return pd.DataFrame(records).sort_values(["model", "seed", "parameter"])


def summarize_importances(long: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for model, model_frame in long.groupby("model", sort=False):
        pivot = model_frame.pivot(
            index="seed", columns="parameter", values="importance"
        ).fillna(0.0)
        top_counts = pivot.idxmax(axis=1).value_counts()
        summary = pd.DataFrame(
            {
                "mean_importance": pivot.mean(),
                "std_importance": pivot.std(ddof=1),
                "se_importance": pivot.sem(ddof=1),
                "n_seeds": pivot.count(),
            }
        )
        summary["top_parameter_count"] = summary.index.map(top_counts).fillna(0).astype(int)
        summary["model"] = model
        summary["model_label"] = MODEL_SPECS[model]["label"]
        frames.append(summary.reset_index(names="parameter"))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["model", "mean_importance"], ascending=[True, False]
    )


def plot_model(summary: pd.DataFrame, model: str, output_dir: Path) -> Path:
    frame = summary[summary["model"] == model].sort_values("mean_importance")
    spec = MODEL_SPECS[model]
    labels = [PARAMETER_LABELS.get(parameter, parameter) for parameter in frame["parameter"]]

    height = max(3.2, 0.48 * len(frame) + 1.35)
    fig, ax = plt.subplots(figsize=(7.2, height))
    ax.barh(
        labels,
        frame["mean_importance"],
        xerr=frame["se_importance"],
        color=spec["color"],
        alpha=0.88,
        error_kw={"ecolor": "#303030", "elinewidth": 1.2, "capsize": 3},
    )
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Mean fANOVA importance")
    ax.xaxis.grid(True, color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)

    for bar, mean, se in zip(
        ax.patches, frame["mean_importance"], frame["se_importance"]
    ):
        ax.text(
            min(mean + se + 0.012, 0.475),
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#303030",
        )

    fig.tight_layout()
    output_path = output_dir / spec["filename"]
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Figure 9 by averaging per-seed Optuna fANOVA importances."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        help="One completed single-seed Figure 9 run directory; repeat for each seed.",
    )
    inputs.add_argument(
        "--run-root",
        type=Path,
        help="Parent directory containing the single-seed Figure 9 run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the three plots and the underlying CSV files.",
    )
    parser.add_argument("--expected-trials", type=int, default=50)
    parser.add_argument("--fanova-seed", type=int, default=0)
    args = parser.parse_args()

    if args.run_root is not None:
        run_root = resolve_path(args.run_root)
        run_dirs = sorted(
            path for path in run_root.glob("*_seed*_trial*") if path.is_dir()
        )
        if not run_dirs:
            raise FileNotFoundError(
                f"No single-seed Figure 9 directories found under {run_root}"
            )
    else:
        run_dirs = [resolve_path(path) for path in args.run_dir]
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    long = read_importances(run_dirs, args.expected_trials, args.fanova_seed)
    summary = summarize_importances(long)
    long_path = output_dir / "figure9_fanova_importance_over_seeds.csv"
    summary_path = output_dir / "figure9_fanova_importance_seed_average.csv"
    long.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)

    outputs = [long_path, summary_path]
    for model in MODEL_SPECS:
        outputs.append(plot_model(summary, model, output_dir))

    print(f"Seeds: {sorted(long['seed'].unique())}")
    print("Wrote outputs:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
