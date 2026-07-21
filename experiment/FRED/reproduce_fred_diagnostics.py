"""Reproduce FRED appendix diagnostics from saved predictions.

This script regenerates the FRED appendix figures and volatility-regime
Diebold-Mariano table from saved ``df_pred.csv`` files. It does not retrain
models. Pass either ``--run-dir`` for a single FRED run containing all models,
or pass all four model-group directories explicitly.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feval import helpers, mgw


REPO_ROOT = Path(__file__).resolve().parents[2]
REPLICATION_ROOT = REPO_ROOT
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostic_outputs"

TARGET_COUNT = 127
# Paper FRED diagnostics use RPI (idx0) for Figure 19 and T10YFFM
# (idx91) for the additional FRED diagnostic figure.
DEFAULT_TARGETS = [0, 91]
REFERENCE_PREFIX = "pCAA_NNOpt_var_"
DM_PREFIXES = [
    "lasso_",
    "fan_fast_",
    "pls_",
    "di_",
    "arp_",
    "vanillaNNOpt_",
    "pcr_",
]

DISPLAY_NAMES = {
    "factorAugmentedNNOpt": "FAR-NN",
    "fan_fast": "FAST-NN",
    "vanillaNNOpt": "vanillaNN",
    "oripCA_NN_PCA_ADDOpt": "PCA_NN_PCA_ADD",
    "pCA_NNOpt_var": "PCA_NN",
    "pCAA_NNOpt_var": "GFANN",
    "gFANNOpt": "GFANN",
    "lasso": "lasso",
    "pcr": "pcr",
    "pls": "pls",
    "di": "di",
    "arp": "arp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute FRED appendix plots and volatility-regime DM table."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Single timestamped FRED run containing all paper models. "
            "If provided, it is used for GFANN, FAST-NN, benchmarks, Lasso, and PCR."
        ),
    )
    parser.add_argument(
        "--gfann-dir",
        type=Path,
        default=None,
        help=(
            "Optional separate directory containing GFANN fred_idx*/df_pred.csv files. "
            "If one separate directory is provided, all four separate directory "
            "arguments must be provided."
        ),
    )
    parser.add_argument(
        "--fast-dir",
        type=Path,
        default=None,
        help="Optional separate directory containing FAST-NN predictions.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Optional separate directory containing benchmark predictions.",
    )
    parser.add_argument(
        "--lasso-pcr-dir",
        type=Path,
        default=None,
        help="Optional separate directory containing Lasso and PCR predictions.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=DEFAULT_TARGETS,
        help=(
            "FRED target indices for figure generation. The default is paper-only: "
            "idx0 (RPI) for Figure 19 and idx91 (T10YFFM) for "
            "fig:fred-pva-more."
        ),
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help=(
            "Do not remove previously generated FRED diagnostic PNG/CSV/TXT files "
            "from output-dir before writing new outputs."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_input_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.run_dir is not None:
        run_dir = resolve_path(args.run_dir)
        return run_dir, run_dir, run_dir, run_dir

    separate_dirs = [
        args.gfann_dir,
        args.fast_dir,
        args.benchmark_dir,
        args.lasso_pcr_dir,
    ]
    if all(path is not None for path in separate_dirs):
        return tuple(resolve_path(path) for path in separate_dirs)  # type: ignore[arg-type]

    raise ValueError(
        "Pass --run-dir, or provide all of --gfann-dir, --fast-dir, "
        "--benchmark-dir, and --lasso-pcr-dir."
    )


def load_fred_scalers_and_dates() -> tuple[np.ndarray, np.ndarray, list[str], pd.Series]:
    raw = np.genfromtxt(
        REPLICATION_ROOT / "data" / "FRED-MD" / "transformed_modern.csv",
        delimiter=",",
    )
    valid_mask = np.isfinite(raw).all(axis=1)
    valid_data = raw[valid_mask]

    # exp_FRED.py uses the first 140 valid rows as the training set.
    train_data = valid_data[:140]
    train_y_mean = train_data.mean(axis=0)
    train_y_std = train_data.std(axis=0)

    modern = pd.read_csv(REPLICATION_ROOT / "data" / "FRED-MD" / "modern.csv")
    fred_names = list(modern.columns[1:])

    # transformed_modern.csv drops two rows relative to the date-bearing modern.csv
    # after the transformation row. This aligns valid index 200 with Oct. 2009.
    data_dates = pd.to_datetime(modern.iloc[1:]["sasdate"], errors="coerce")
    valid_raw_positions = np.where(valid_mask)[0]
    valid_dates = data_dates.iloc[valid_raw_positions - 2].reset_index(drop=True)
    return train_y_mean, train_y_std, fred_names, valid_dates


def parse_target_idx(folder_name: str) -> int:
    match = re.search(r"fred_idx(\d+)_trial", folder_name)
    if match is None:
        raise ValueError(f"Cannot parse FRED target index from {folder_name}")
    return int(match.group(1))


def rename_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if col.startswith("gFANNOpt_"):
            rename[col] = col.replace("gFANNOpt_", "pCAA_NNOpt_var_", 1)
    return df.rename(columns=rename)


def first_col(df: pd.DataFrame, prefix: str) -> str:
    matches = [col for col in df.columns if col.startswith(prefix)]
    if not matches:
        raise KeyError(f"No column starts with {prefix}")
    return matches[0]


def read_predictions(
    folder: Path,
    train_y_mean: np.ndarray,
    train_y_std: np.ndarray,
    valid_dates: pd.Series,
) -> dict[int, pd.DataFrame]:
    out = {}
    for child in sorted(folder.iterdir()):
        pred_path = child / "df_pred.csv"
        if not pred_path.exists():
            continue
        idx = parse_target_idx(child.name)
        df = pd.read_csv(pred_path, index_col=0).T
        pred_cols = [col for col in df.columns if col not in {"y", "date"}]
        df.loc[:, pred_cols] = df[pred_cols] * train_y_std[idx] + train_y_mean[idx]
        df = rename_prediction_columns(df)

        date_idx = pd.to_numeric(df["date"], errors="coerce").astype(int)
        df["date"] = valid_dates.iloc[date_idx].to_numpy()
        out[idx] = df.set_index("date")

    if len(out) != TARGET_COUNT:
        raise ValueError(f"Expected {TARGET_COUNT} prediction files in {folder}, found {len(out)}")
    return out


def build_merged_predictions(
    gfann_dir: Path,
    fast_dir: Path,
    benchmark_dir: Path,
    lasso_pcr_dir: Path,
) -> tuple[dict[int, pd.DataFrame], list[str]]:
    train_y_mean, train_y_std, fred_names, valid_dates = load_fred_scalers_and_dates()

    pred_gfann = read_predictions(
        gfann_dir, train_y_mean, train_y_std, valid_dates
    )
    pred_fast = read_predictions(
        fast_dir, train_y_mean, train_y_std, valid_dates
    )
    pred_benchmark = read_predictions(
        benchmark_dir, train_y_mean, train_y_std, valid_dates
    )
    pred_lasso_pcr = read_predictions(
        lasso_pcr_dir, train_y_mean, train_y_std, valid_dates
    )

    merged = {}
    for idx in range(TARGET_COUNT):
        gf = pred_gfann[idx]
        fast = pred_fast[idx]
        bm = pred_benchmark[idx]
        lp = pred_lasso_pcr[idx]

        merged[idx] = pd.concat(
            [
                gf[[first_col(gf, REFERENCE_PREFIX)]],
                lp[[first_col(lp, "lasso_")]],
                fast[[first_col(fast, "fan_fast_")]],
                bm[[first_col(bm, "pls_")]],
                bm[[first_col(bm, "di_")]],
                bm[[first_col(bm, "arp_")]],
                bm[[first_col(bm, "vanillaNNOpt_")]],
                lp[[first_col(lp, "pcr_")]],
                gf[["y"]],
            ],
            axis=1,
        ).dropna()
    return merged, fred_names


def model_display_name(col: str) -> str:
    base = col.split("_140")[0]
    return DISPLAY_NAMES.get(base, base)


def calibration_curve_and_table(df: pd.DataFrame, n_bins: int = 10) -> tuple[pd.DataFrame, plt.Figure]:
    yhat = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    mask = np.isfinite(yhat) & np.isfinite(y)
    yhat, y = yhat[mask], y[mask]

    q = pd.qcut(yhat, q=n_bins, labels=False, duplicates="drop")
    tab = (
        pd.DataFrame({"yhat": yhat, "y": y, "bin": q})
        .groupby("bin", observed=True)
        .agg(count=("y", "size"), pred_mean=("yhat", "mean"), actual_mean=("y", "mean"))
        .reset_index(drop=True)
    )
    tab.insert(0, "bin", np.arange(1, len(tab) + 1))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(tab["pred_mean"], tab["actual_mean"], marker="o", lw=2, label="Bin means")
    lim_min = tab[["pred_mean", "actual_mean"]].min().min()
    lim_max = tab[["pred_mean", "actual_mean"]].max().max()
    pad = 0.05 * (lim_max - lim_min) if lim_max > lim_min else 1.0
    lims = (lim_min - pad, lim_max + pad)
    ax.plot(lims, lims, lw=2, label="Perfect calibration (y=x)")
    ax.set_xlabel("Average prediction (per bin)")
    ax.set_ylabel("Average actual (per bin)")
    ax.set_title("Calibration Curve (prediction-binned)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return tab, fig


def plot_target_diagnostics(
    merged: dict[int, pd.DataFrame],
    fred_names: list[str],
    target_idx: int,
    output_dir: Path,
) -> tuple[list[Path], list[dict[str, object]]]:
    output_paths = []
    records = []
    df_all = merged[target_idx]
    for prefix in ("lasso_", REFERENCE_PREFIX):
        model_col = first_col(df_all, prefix)
        model_base = model_col.split("_140")[0]
        label = model_display_name(model_col)
        df_raw = df_all[[model_col, "y"]].copy()
        df_raw.columns = ["yhat", "y"]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_raw.index, df_raw["y"], label="Actual", linewidth=2)
        ax.plot(df_raw.index, df_raw["yhat"], label="Predicted", linestyle="-.")
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(f"Predicted vs Actual Over Time ({label}, target={fred_names[target_idx]})")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Target variable {fred_names[target_idx]}")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        out_path = output_dir / f"FRED_pred_vs_actual_{model_base}_idx{target_idx}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)
        records.append(
            {
                "paper_label": paper_labels_for(target_idx, "predicted_vs_actual"),
                "panel": "(a)" if prefix == "lasso_" else "(b)",
                "target_idx": target_idx,
                "target_name": fred_names[target_idx],
                "model": label,
                "diagnostic": "predicted_vs_actual",
                "file": out_path.name,
            }
        )

        _, fig_cal = calibration_curve_and_table(df_raw[["yhat", "y"]], n_bins=10)
        out_path = output_dir / f"FRED_bin_calibration_{model_base}_idx{target_idx}.png"
        fig_cal.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig_cal)
        output_paths.append(out_path)
        records.append(
            {
                "paper_label": paper_labels_for(target_idx, "bin_calibration"),
                "panel": "(a)" if prefix == "lasso_" else "(b)",
                "target_idx": target_idx,
                "target_name": fred_names[target_idx],
                "model": label,
                "diagnostic": "bin_calibration",
                "file": out_path.name,
            }
        )
    return output_paths, records


def paper_labels_for(target_idx: int, diagnostic: str) -> str:
    labels = []
    if target_idx == 0 and diagnostic == "predicted_vs_actual":
        labels.append("fig:FRED_pred_vs_actual")
    if target_idx == 0 and diagnostic == "bin_calibration":
        labels.append("fig:FRED_bin_calibration")
    if target_idx == 91:
        labels.append("fig:fred-pva-more")
    return ";".join(labels)


def clean_diagnostic_outputs(output_dir: Path) -> None:
    for pattern in [
        "FRED_pred_vs_actual_*.png",
        "FRED_bin_calibration_*.png",
        "FRED_dm_regimes.csv",
        "FRED_dm_regimes_latex_rows.txt",
        "FRED_diagnostic_manifest.csv",
    ]:
        for path in output_dir.glob(pattern):
            path.unlink()


def dm_signed_t(loss_matrix: np.ndarray, bw: int = 0) -> float:
    d = (loss_matrix[:, 0] - loss_matrix[:, 1]).astype(float)
    if not np.isfinite(d).all():
        return np.nan
    stat, _, _ = mgw(loss_matrix, bw=bw)
    return float(np.sign(d.mean()) * np.sqrt(stat))


def compute_dm_table(
    merged: dict[int, pd.DataFrame],
    h: int = 1,
    window: int = 20,
    quantile: float = 0.7,
    bw: int = 0,
) -> pd.DataFrame:
    res_all, res_hi, res_lo = {}, {}, {}

    for idx, df in merged.items():
        pred = df[[col for col in df.columns if col != "y"]]
        yv = df["y"].to_numpy().reshape(-1, 1)

        base_valid = np.isfinite(yv).ravel() & np.isfinite(pred.values).all(axis=1)
        y_base = yv[base_valid]
        pred_base = pred.iloc[base_valid].reset_index(drop=True)

        sigma = df["y"].shift(h).rolling(window, min_periods=window // 2).std()
        q_hi = sigma.rolling(window, min_periods=window // 3).quantile(quantile)
        q_lo = sigma.rolling(window, min_periods=window // 3).quantile(1.0 - quantile)
        high_vol = (sigma >= q_hi).fillna(False).to_numpy()[base_valid].astype(bool)
        low_vol = (sigma <= q_lo).fillna(False).to_numpy()[base_valid].astype(bool)

        out_all, out_hi, out_lo = {}, {}, {}
        for j in range(1, pred_base.shape[1]):
            name = pred_base.columns[j]
            loss = helpers.se(y_base, pred_base.iloc[:, [0, j]])
            loss_np = loss.values if hasattr(loss, "values") else np.asarray(loss)
            out_all[name] = dm_signed_t(loss_np, bw=bw)
            out_hi[name] = dm_signed_t(loss_np[high_vol], bw=bw) if high_vol.sum() >= 3 else np.nan
            out_lo[name] = dm_signed_t(loss_np[low_vol], bw=bw) if low_vol.sum() >= 3 else np.nan

        res_all[idx] = out_all
        res_hi[idx] = out_hi
        res_lo[idx] = out_lo

    res_dic = {
        "all": pd.DataFrame(res_all).T.mean(),
        "high_vol": pd.DataFrame(res_hi).T.mean(),
        "low_vol": pd.DataFrame(res_lo).T.mean(),
    }
    dm_table = pd.DataFrame(res_dic).T.round(3)
    dm_table.columns = [model_display_name(col) for col in dm_table.columns]
    dm_table = dm_table[["lasso", "FAST-NN", "pls", "di", "arp", "vanillaNN", "pcr"]]
    dm_table.index.name = "regime"
    return dm_table


def latex_dm_rows(dm_table: pd.DataFrame) -> str:
    header = "Regime & " + " & ".join(dm_table.columns) + r" \\ \hline"
    rows = [header]
    for idx, row in dm_table.iterrows():
        values = " & ".join(f"{value:.3f}" for value in row)
        rows.append(f"{idx} & {values} " + r"\\ \hline")
    return "\n".join(rows)


def run_diagnostics(
    gfann_dir: Path,
    fast_dir: Path,
    benchmark_dir: Path,
    lasso_pcr_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    targets: list[int] | None = None,
    clean_output: bool = True,
) -> dict[str, object]:
    if targets is None:
        targets = DEFAULT_TARGETS
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        clean_diagnostic_outputs(output_dir)

    merged, fred_names = build_merged_predictions(
        gfann_dir, fast_dir, benchmark_dir, lasso_pcr_dir
    )
    figure_paths = []
    figure_records = []
    for target_idx in targets:
        paths, records = plot_target_diagnostics(merged, fred_names, target_idx, output_dir)
        figure_paths.extend(paths)
        figure_records.extend(records)

    manifest_path = output_dir / "FRED_diagnostic_manifest.csv"
    pd.DataFrame(figure_records).to_csv(manifest_path, index=False)

    dm_table = compute_dm_table(merged)
    dm_table_path = output_dir / "FRED_dm_regimes.csv"
    dm_table.to_csv(dm_table_path)
    (output_dir / "FRED_dm_regimes_latex_rows.txt").write_text(latex_dm_rows(dm_table) + "\n")

    return {
        "merged": merged,
        "fred_names": fred_names,
        "figure_paths": figure_paths,
        "manifest_path": manifest_path,
        "dm_table": dm_table,
        "output_dir": output_dir,
    }


def main() -> None:
    args = parse_args()
    gfann_dir, fast_dir, benchmark_dir, lasso_pcr_dir = resolve_input_dirs(args)

    out = run_diagnostics(
        gfann_dir=gfann_dir,
        fast_dir=fast_dir,
        benchmark_dir=benchmark_dir,
        lasso_pcr_dir=lasso_pcr_dir,
        output_dir=args.output_dir,
        targets=args.targets,
        clean_output=not args.no_clean_output,
    )
    print(f"Wrote FRED diagnostics to: {out['output_dir']}")
    print(f"Wrote figure manifest to: {out['manifest_path']}")
    print(out["dm_table"])


if __name__ == "__main__":
    main()
