"""Reproduce the ETF trading-performance tables from saved predictions.

This script recomputes the two forecast-to-position rules reported in the
paper's ETF section from the saved ``df_pred.csv`` files. It does not retrain
any models.

By default, the script uses the latest ETF result directory under
``logs/ETF`` that contains ``model_*/df_pred.csv`` files. You can
also pass ``--selected-dir`` explicitly. The paper tables do not use warm-up
predictions.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
REPLICATION_ROOT = REPO_ROOT
ETF_LOG_ROOT = REPLICATION_ROOT / "logs" / "ETF"

sys.path.insert(0, str(REPLICATION_ROOT))
from methods.stat_methods_soft import results_analytics_  # noqa: E402


METRIC_COLUMNS = [
    "annualized_return",
    "sharpe_ratio",
    "pct_max_dd",
    "turnover",
    "dir_accuracy",
    "IC",
    "avg_pct_pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute ETF Tables 3 and 4 metrics from saved model predictions."
        )
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing model_*/df_pred.csv files. If omitted, "
            "the latest matching ETF result directory under logs/ETF "
            "is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for regenerated CSVs. Defaults to --selected-dir, "
            "overwriting the table CSVs there."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=126,
        help="Rolling winsorization window used by results_analytics_.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if a prediction row and y row have different lengths.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def find_latest_etf_result_dir(log_root: Path = ETF_LOG_ROOT) -> Path:
    if not log_root.exists():
        raise FileNotFoundError(
            f"ETF log root does not exist: {log_root}. "
            "Run experiment/ETF/run_scripts_ETF.py first, "
            "or pass --selected-dir."
        )
    candidates = [
        p
        for p in log_root.rglob("*")
        if p.is_dir() and any(p.glob("model_*/df_pred.csv"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No ETF result directory with model_*/df_pred.csv found under {log_root}. "
            "Run experiment/ETF/run_scripts_ETF.py first, "
            "or pass --selected-dir."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_selected_dir(selected_dir: Path | None) -> Path:
    if selected_dir is not None:
        return resolve_path(selected_dir)
    return find_latest_etf_result_dir()


def read_prediction_block(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    index_col = df.columns[0]
    return df.set_index(index_col)


def prediction_rows(df: pd.DataFrame) -> list[str]:
    return [idx for idx in df.index if idx not in {"y", "date"}]


def count_prediction_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(prediction_rows(read_prediction_block(path)))


def row_to_column(df: pd.DataFrame, row_name: str) -> np.ndarray:
    values = pd.to_numeric(df.loc[row_name], errors="coerce").dropna().to_numpy()
    return values.reshape(-1, 1)


def parse_seed(row_name: str) -> int:
    match = re.search(r"_(\d+)$", row_name)
    if match is None:
        raise ValueError(f"Cannot parse seed from prediction row: {row_name}")
    return int(match.group(1))


def align_signal_and_y(
    signal: np.ndarray, y: np.ndarray, *, strict: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if len(signal) == len(y):
        return signal, y
    if strict:
        raise ValueError(
            f"Prediction length {len(signal)} does not match y length {len(y)}."
        )
    min_len = min(len(signal), len(y))
    return signal[-min_len:], y[-min_len:]


def extract_suffix_metrics(result: pd.DataFrame, suffix: str) -> dict[str, float]:
    row = {}
    for metric in METRIC_COLUMNS:
        row[metric] = float(result[f"{metric}{suffix}"].iloc[0])
    return row


def iter_model_dirs(selected_dir: Path) -> Iterable[Path]:
    yield from sorted(
        p for p in selected_dir.iterdir() if p.is_dir() and p.name.startswith("model_")
    )


def compute_metrics(selected_dir: Path, *, window: int, strict: bool) -> pd.DataFrame:
    rows = []
    selected_dir = selected_dir.resolve()

    for model_dir in iter_model_dirs(selected_dir):
        pred_path = model_dir / "df_pred.csv"
        if not pred_path.exists():
            continue

        pred_df = read_prediction_block(pred_path)
        pred_rows = prediction_rows(pred_df)
        y = row_to_column(pred_df, "y")
        n_pred_rows = len(pred_rows)
        n_warmup_pred_rows = count_prediction_rows(model_dir / "df_pred_warmup.csv")
        model = model_dir.name.removeprefix("model_")

        for pred_row in pred_rows:
            signal = row_to_column(pred_df, pred_row)
            signal, y_aligned = align_signal_and_y(signal, y, strict=strict)
            result = results_analytics_(signal, y_aligned, window=window)
            seed = parse_seed(pred_row)

            for suffix in ("1", "2"):
                row = extract_suffix_metrics(result, suffix)
                row.update(
                    {
                        "model_dir_name": model_dir.name,
                        "model": model,
                        "seed": seed,
                        "pred_row": pred_row,
                        "warmup_mode_used": "no_warmup",
                        "warmup_exact_seed_available": False,
                        "n_pred_rows_in_model": n_pred_rows,
                        "n_warmup_pred_rows_in_model": n_warmup_pred_rows,
                        "source_dir": str(selected_dir),
                        "suffix": int(suffix),
                        "window": window,
                        "scale_window": 126,
                        "target_volatility": 0.20,
                        "rolling_window": 5,
                    }
                )
                rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No model_*/df_pred.csv files found in {selected_dir}")
    return pd.DataFrame(rows)


def seed_average(over_seed_df: pd.DataFrame, suffix: int) -> pd.DataFrame:
    df = over_seed_df[over_seed_df["suffix"] == suffix].copy()
    group_cols = ["model", "model_dir_name"]
    metric_mean = df.groupby(group_cols, sort=False)[METRIC_COLUMNS].mean()
    meta = df.groupby(group_cols, sort=False).agg(
        suffix=("suffix", "first"),
        warmup_mode_used=("warmup_mode_used", "first"),
        n_seeds=("seed", "nunique"),
        seed_min=("seed", "min"),
        seed_max=("seed", "max"),
        window=("window", "first"),
        scale_window=("scale_window", "first"),
        rolling_window=("rolling_window", "first"),
        target_volatility=("target_volatility", "first"),
        n_pred_rows_in_model=("n_pred_rows_in_model", "first"),
        n_warmup_pred_rows_in_model=("n_warmup_pred_rows_in_model", "first"),
    )
    averaged = pd.concat([meta, metric_mean], axis=1).reset_index()
    return averaged[
        [
            "model",
            "model_dir_name",
            "suffix",
            "warmup_mode_used",
            "n_seeds",
            "seed_min",
            "seed_max",
            *METRIC_COLUMNS,
            "window",
            "scale_window",
            "rolling_window",
            "target_volatility",
            "n_pred_rows_in_model",
            "n_warmup_pred_rows_in_model",
        ]
    ]


def write_outputs(over_seed_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (1, 2):
        over_seed_suffix = over_seed_df[over_seed_df["suffix"] == suffix].copy()
        over_seed_suffix.to_csv(
            output_dir / f"suffix{suffix}_metrics_over_seeds.csv", index=False
        )
        seed_average(over_seed_df, suffix).to_csv(
            output_dir / f"suffix{suffix}_metrics_seed_average.csv", index=False
        )


def main() -> None:
    args = parse_args()
    selected_dir = resolve_selected_dir(args.selected_dir)
    output_dir = args.output_dir or selected_dir

    over_seed_df = compute_metrics(
        selected_dir, window=args.window, strict=args.strict
    )
    write_outputs(over_seed_df, output_dir)
    print(f"Wrote ETF table metrics to: {output_dir}")


if __name__ == "__main__":
    main()
