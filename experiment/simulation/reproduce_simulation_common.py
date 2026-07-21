from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "Simulation"
TABLE_OUT = SIM_DIR / "table_outputs"
FIG_OUT = SIM_DIR / "figure_outputs"

DATASET_MAP = {(0, 2): "DS11", (1, 2): "DS21"}

MAIN_MODEL_MAP = {
    "oracleNNOpt": "oracleNN",
    "fAR-NNOpt": "FAR-NN",
    "vanillaNNOpt": "vanillaNN",
    "autoencoderOpt": "autoencoder",
    "lasso": "lasso",
    "pcr": "pcr",
    "pls": "pls",
    "di": "di",
    "arp": "arp",
    "pCA_NN_PCA_ADDOpt": "PCA_NN_PCA_ADD",
    "sPCA_NNOpt": "SPCA_NN",
    "nN_SPCA_NNOpt": "NN_SPCA_NN",
    "sPCA_NN_SPCA_ADDOpt": "SPCA_NN_SPCA_ADD",
}

MODEL_COLORS = {
    "FAR-NN": "#1f77b4",
    "oracleNN": "#7f7f7f",
    "vanillaNN": "#8c564b",
    "autoencoder": "#e377c2",
    "lasso": "#bcbd22",
    "pcr": "#17becf",
    "pls": "#9edae5",
    "di": "#c7c7c7",
    "arp": "#dbdb8d",
    "SPCA_NN": "#2ca02c",
    "NN_SPCA_NN": "#98df8a",
    "PCA_NN_PCA_ADD": "#ff7f0e",
    "SPCA_NN_SPCA_ADD": "#d62728",
    "PCA_NN_ADD_PCA": "#ffbb78",
    "PCA_NN_PCA": "#9467bd",
    "SPCA_NN_ADD_SPCA": "#ff9896",
    "SPCA_NN_SPCA": "#c5b0d5",
    "Joint-SPCA_NN": "#1f77b4",
    "Frozen-SPCA_NN": "#2ca02c",
    "SPCA_NN (frozen)": "#2ca02c",
}


def ensure_output_dirs() -> None:
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)


def normalize_patterns(pattern: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(pattern, str):
        return [pattern]
    return list(pattern)


def matches_any_pattern(name: str, pattern: str | list[str] | tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in normalize_patterns(pattern))


def format_patterns(pattern: str | list[str] | tuple[str, ...]) -> str:
    return ", ".join(repr(pat) for pat in normalize_patterns(pattern))


def add_common_log_args(
    parser: argparse.ArgumentParser,
    *,
    pattern: str | list[str] | tuple[str, ...],
    run_help: str,
) -> None:
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help="Root directory containing Simulation run logs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=run_help,
    )
    parser.set_defaults(auto_pattern=pattern)


def resolve_run_dir(log_root: Path, run_dir: Path | None, pattern: str | list[str] | tuple[str, ...]) -> Path:
    if run_dir is not None:
        out = run_dir
        if not out.is_absolute():
            out = REPO_ROOT / out
        if not out.exists():
            raise FileNotFoundError(f"Run directory does not exist: {out}")
        if any(out.glob("p_*/seed*/summary_file.csv")):
            return out
        nested = [
            p for p in out.iterdir()
            if p.is_dir()
            and matches_any_pattern(p.name, pattern)
            and any(p.glob("p_*/seed*/summary_file.csv"))
        ]
        if len(nested) == 1:
            return nested[0]
        if len(nested) > 1:
            raise ValueError(
                f"Run directory {out} contains multiple matching nested runs: {nested}. "
                "Please pass one nested run directory explicitly."
            )
        return out

    candidates = [
        p for p in log_root.iterdir()
        if p.is_dir() and matches_any_pattern(p.name, pattern)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run directory matching {format_patterns(pattern)} found under {log_root}. "
            "Run the corresponding run_scripts_*.py first, or pass --run-dir."
        )
    return sorted(candidates, key=lambda p: p.name)[-1]


def parse_p_seed(path: Path) -> tuple[int, int]:
    return int(path.parents[1].name.split("_")[1]), int(path.parent.name.replace("seed", ""))


def test_col(raw_model: str) -> str:
    return raw_model if raw_model.endswith("_test") else f"{raw_model}_test"


def read_model_tests(
    run_dir: Path,
    model_map: dict[str, str],
    *,
    dataset_map: dict[tuple[int, int], str] = DATASET_MAP,
    row_filter: Callable[[pd.Series], bool] | None = None,
    extra_fields: Callable[[pd.Series], dict] | None = None,
) -> pd.DataFrame:
    rows = []
    paths = sorted(run_dir.glob("p_*/seed*/summary_file.csv"))
    if not paths:
        raise FileNotFoundError(f"No summary_file.csv files found under {run_dir}")

    for path in paths:
        p, seed = parse_p_seed(path)
        df = pd.read_csv(path)
        df = df[pd.to_numeric(df["factor_id"], errors="coerce").notna()]
        for _, row in df.iterrows():
            key = (int(row["factor_id"]), int(row["hcm_id"]))
            dataset = dataset_map.get(key)
            if dataset is None:
                continue
            if row_filter is not None and not row_filter(row):
                continue
            extras = extra_fields(row) if extra_fields is not None else {}
            for raw_model, model in model_map.items():
                col = test_col(raw_model)
                if col not in df.columns or pd.isna(row[col]):
                    continue
                rows.append({
                    "dataset": dataset,
                    "p": p,
                    "seed": seed,
                    "factor_id": key[0],
                    "hcm_id": key[1],
                    "model": model,
                    "test_mse": float(row[col]),
                    "source_file": str(path),
                    **extras,
                })

    if not rows:
        raise ValueError(f"No usable model test rows found under {run_dir}")
    return pd.DataFrame(rows).sort_values(["dataset", "p", "seed", "model"]).reset_index(drop=True)


def aggregate_mean_and_counts(df: pd.DataFrame, index_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = df.groupby(index_cols + ["model"])["test_mse"].mean().unstack("model").sort_index()
    counts = df.groupby(index_cols + ["model"])["seed"].nunique().unstack("model").sort_index()
    return mean, counts


def plot_lines(
    table: pd.DataFrame,
    cols: list[str],
    *,
    title: str,
    output_path: Path,
    ylabel: str = "test MSE",
) -> None:
    plot_cols = [c for c in cols if c in table.columns]
    if not plot_cols:
        raise ValueError(f"None of the requested columns are present: {cols}")

    fallback_cmap = plt.get_cmap("tab20", max(len(plot_cols), 1))
    colors = {
        model: MODEL_COLORS.get(model, fallback_cmap(i))
        for i, model in enumerate(plot_cols)
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in plot_cols:
        series = table[model].dropna()
        ax.plot(series.index, series.values, marker="o", linewidth=2, label=model, color=colors[model])
    ax.set_title(title)
    ax.set_xlabel("dimension p")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_table_and_figure_notice(paths: list[Path]) -> None:
    print("Wrote outputs:")
    for path in paths:
        print(f"  {path}")
