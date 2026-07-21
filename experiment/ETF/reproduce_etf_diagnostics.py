"""Reproduce ETF appendix diagnostics from saved predictions.

This script regenerates the ETF-related appendix artifacts:

* predicted-vs-actual figures for lasso and PCA_NN_PCA_ADD,
* standardized predicted-vs-actual diagnostic figures for the same two models,
* prediction-binned calibration figures for the same two models, and
* the volatility-regime winsorized-forecast Diebold-Mariano signed
  t-statistic table.

The script uses saved ``df_pred.csv`` files from an ETF result directory and
does not retrain models. By default, it uses the latest ETF result directory
under ``logs/ETF`` that contains ``model_*/df_pred.csv`` files, and
seed 100 for the neural-network forecast.
"""

from __future__ import annotations

import argparse
import re
import sys
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
ETF_LOG_ROOT = REPLICATION_ROOT / "logs" / "ETF"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostic_outputs"


REFERENCE_MODEL = "pCA_NN_PCA_ADDOpt"
FIGURE_MODELS = ["lasso", REFERENCE_MODEL]
DM_BENCHMARKS = ["lasso", "pls", "arp", "di", "ewma", "pcr"]
DM_WINSOR_WINDOW = 126
DM_WINSOR_THRESHOLD = 0.05

DISPLAY_NAMES = {
    "pCA_NN_PCA_ADDOpt": "PCA_NN_PCA_ADD",
    "sPCA_NN_SPCA_ADDOpt": "SPCA_NN_SPCA_ADD",
    "fAR-NNOpt": "FAR-NN",
    "vanillaNNOpt": "vanillaNN",
    "autoencoderOpt": "autoencoder",
}


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


def parse_seed(row_name: str) -> int | None:
    match = re.search(r"_(\d+)$", row_name)
    return None if match is None else int(match.group(1))


def choose_prediction_row(rows: list[str], seed: int) -> str:
    exact = [row for row in rows if parse_seed(row) == seed]
    if exact:
        return exact[-1]
    if len(rows) == 1:
        return rows[0]
    raise ValueError(f"Could not find seed {seed}; available rows are {rows[:5]}...")


def load_model_forecast(
    selected_dir: Path,
    model: str,
    seed: int,
    model_dir_overrides: dict[str, Path] | None = None,
) -> pd.DataFrame:
    if model_dir_overrides is not None and model in model_dir_overrides:
        pred_path = Path(model_dir_overrides[model]) / "df_pred.csv"
    else:
        pred_path = selected_dir / f"model_{model}" / "df_pred.csv"
    pred_df = read_prediction_block(pred_path)
    pred_row = choose_prediction_row(prediction_rows(pred_df), seed)

    y = pd.to_numeric(pred_df.loc["y"], errors="coerce")
    yhat = pd.to_numeric(pred_df.loc[pred_row], errors="coerce")
    if "date" in pred_df.index:
        date = pd.to_datetime(pred_df.loc["date"], errors="coerce")
    else:
        date = pd.RangeIndex(len(y))

    out = pd.DataFrame({"date": date, model: yhat, "y": y}).dropna()
    out = out.set_index("date")
    return out


def load_forecast_matrix(
    selected_dir: Path,
    models: list[str],
    seed: int,
    model_dir_overrides: dict[str, Path] | None = None,
) -> pd.DataFrame:
    frames = []
    y_series = None
    for model in models:
        frame = load_model_forecast(
            selected_dir, model, seed, model_dir_overrides=model_dir_overrides
        )
        frames.append(frame[[model]])
        if y_series is None:
            y_series = frame["y"]
    df = pd.concat(frames + [y_series.rename("y")], axis=1).dropna()
    return df


def rolling_calibrate_df(
    df: pd.DataFrame,
    window: int = 252,
    step: int = 60,
    intercept: bool = True,
    min_obs: int | None = None,
) -> pd.DataFrame:
    """Calibrate forecasts with a trailing-window linear mapping."""

    if df.shape[1] < 2:
        raise ValueError("df must have at least two columns: [forecast, target].")
    if min_obs is None:
        min_obs = max(20, window // 5)

    yhat = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    n = len(df)

    yhat_cal = np.full(n, np.nan, dtype=float)
    a_hist = np.full(n, np.nan, dtype=float)
    b_hist = np.full(n, np.nan, dtype=float)
    have_params = False
    a_curr, b_curr = np.nan, np.nan

    for t in range(n):
        if t >= window and ((t - window) % step == 0):
            y_win = y[t - window : t]
            p_win = yhat[t - window : t]
            mask = np.isfinite(y_win) & np.isfinite(p_win)
            if mask.sum() >= max(2, min_obs):
                yw = y_win[mask]
                pw = p_win[mask]
                if intercept:
                    x = np.column_stack([np.ones_like(pw), pw])
                    coef, *_ = np.linalg.lstsq(x, yw, rcond=None)
                    a_curr, b_curr = float(coef[0]), float(coef[1])
                    have_params = True
                else:
                    denom = float(np.dot(pw, pw))
                    if denom > 0:
                        a_curr = 0.0
                        b_curr = float(np.dot(pw, yw) / denom)
                        have_params = True

        if have_params and np.isfinite(yhat[t]):
            yhat_cal[t] = a_curr + b_curr * yhat[t]
        a_hist[t] = a_curr
        b_hist[t] = b_curr

    return pd.DataFrame(
        {"yhat_cal": yhat_cal, "calib_a": a_hist, "calib_b": b_hist, "y": y},
        index=df.index,
    )


def calibration_curve_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    yhat = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    mask = np.isfinite(yhat) & np.isfinite(y)
    data = pd.DataFrame({"yhat": yhat[mask], "y": y[mask]})
    data["bin"] = pd.qcut(data["yhat"], q=n_bins, labels=False, duplicates="drop")
    tab = (
        data.groupby("bin", observed=True)
        .agg(
            count=("y", "size"),
            pred_mean=("yhat", "mean"),
            actual_mean=("y", "mean"),
        )
        .sort_values("pred_mean")
        .reset_index(drop=True)
    )
    tab.insert(0, "bin", np.arange(1, len(tab) + 1))
    return tab


def plot_pred_vs_actual(calibrated: pd.DataFrame, model: str, output_dir: Path) -> Path:
    plot_df = calibrated.iloc[252 * 2 : 252 * 3].dropna()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_df.index, plot_df["y"], label="Actual", linewidth=2)
    ax.plot(plot_df.index, plot_df["yhat_cal"], label="Predicted", linestyle="-.")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title("Predicted vs Actual Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Next-day S&P 500 price-index return")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = output_dir / f"ETF_pred_vs_actual_{model}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def standardize_series(series: pd.Series) -> pd.Series:
    std = series.std()
    if not np.isfinite(std) or std == 0:
        return series * np.nan
    return (series - series.mean()) / std


def plot_standardized_pred_vs_actual(
    calibrated: pd.DataFrame, model: str, output_dir: Path
) -> Path:
    """Plot standardized actual and calibrated forecasts on a common z-score scale."""

    plot_df = calibrated.iloc[252 * 2 : 252 * 3].dropna().copy()
    plot_df["actual_z"] = standardize_series(plot_df["y"])
    plot_df["predicted_z"] = standardize_series(plot_df["yhat_cal"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_df.index, plot_df["actual_z"], label="Actual (standardized)", linewidth=2)
    ax.plot(
        plot_df.index,
        plot_df["predicted_z"],
        label="Predicted (standardized)",
        linestyle="-.",
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title("Standardized Predicted vs Actual Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Z-score within the plotted window")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = output_dir / f"ETF_pred_vs_actual_standardized_{model}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_calibration(calibrated: pd.DataFrame, model: str, output_dir: Path) -> Path:
    plot_df = calibrated[["yhat_cal", "y"]].dropna()
    tab = calibration_curve_table(plot_df, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        tab["pred_mean"],
        tab["actual_mean"],
        marker="o",
        lw=2,
        label="Bin means",
        color="#1f77b4",
    )
    lim_min = tab[["pred_mean", "actual_mean"]].min().min()
    lim_max = tab[["pred_mean", "actual_mean"]].max().max()
    pad = 0.05 * (lim_max - lim_min) if lim_max > lim_min else 1.0
    lims = (lim_min - pad, lim_max + pad)
    ax.plot(lims, lims, lw=2, label="Perfect calibration (y=x)", color="#ff7f0e")
    ax.set_xlabel("Average prediction (per bin)")
    ax.set_ylabel("Average actual (per bin)")
    ax.set_title("Calibration Curve (prediction-binned)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / f"ETF_bin_calibration_{model}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_calibration_diagnostics(
    calibrated: pd.DataFrame, model: str, output_dir: Path, n_bins: int = 10
) -> tuple[Path, Path]:
    """Save bin-level calibration data and the highest-prediction bin observations."""

    plot_df = calibrated[["yhat_cal", "y"]].dropna().copy()
    plot_df["bin"] = pd.qcut(
        plot_df["yhat_cal"], q=n_bins, labels=False, duplicates="drop"
    )
    bin_table = (
        plot_df.groupby("bin", observed=True)
        .agg(
            count=("y", "size"),
            pred_mean=("yhat_cal", "mean"),
            actual_mean=("y", "mean"),
            y_min=("y", "min"),
            y_max=("y", "max"),
            y_std=("y", "std"),
            pred_min=("yhat_cal", "min"),
            pred_max=("yhat_cal", "max"),
        )
        .reset_index()
    )
    bin_table["bin"] = bin_table["bin"].astype(int) + 1

    bin_path = output_dir / f"ETF_bin_calibration_table_{model}.csv"
    bin_table.to_csv(bin_path, index=False)

    highest_bin = plot_df["bin"].max()
    high_bin_df = plot_df.loc[plot_df["bin"].eq(highest_bin), ["yhat_cal", "y"]].copy()
    high_bin_df.insert(0, "date", high_bin_df.index)
    high_bin_df["bin"] = int(highest_bin) + 1
    high_bin_path = output_dir / f"ETF_bin_calibration_highest_bin_{model}.csv"
    high_bin_df.to_csv(high_bin_path, index=False)

    return bin_path, high_bin_path


def dm_signed_t(loss_matrix: np.ndarray, bw: int = 0) -> float:
    d = (loss_matrix[:, 0] - loss_matrix[:, 1]).astype(float)
    if not np.isfinite(d).all():
        return np.nan
    stat, _, _ = mgw(loss_matrix, bw=bw)
    return float(np.sign(d.mean()) * np.sqrt(stat))


def rolling_winsorize_frame(
    df: pd.DataFrame,
    *,
    window: int = DM_WINSOR_WINDOW,
    threshold: float = DM_WINSOR_THRESHOLD,
) -> pd.DataFrame:
    values = df.astype(float)
    upper = values.rolling(window=window, min_periods=1).quantile(1.0 - threshold)
    lower = values.rolling(window=window, min_periods=1).quantile(threshold)
    values = values.mask((values > upper).fillna(False), upper)
    values = values.mask((values < lower).fillna(False), lower)
    return values


def winsorize_forecast_columns(
    forecast_df: pd.DataFrame,
    forecast_columns: list[str],
    *,
    window: int = DM_WINSOR_WINDOW,
    threshold: float = DM_WINSOR_THRESHOLD,
) -> pd.DataFrame:
    df = forecast_df.copy()
    df[forecast_columns] = rolling_winsorize_frame(
        df[forecast_columns], window=window, threshold=threshold
    )
    return df


def compute_dm_table(
    forecast_df: pd.DataFrame,
    reference_model: str = REFERENCE_MODEL,
    benchmarks: list[str] | None = None,
    h: int = 1,
    vol_window: int = 60,
    quantile: float = 0.7,
    bw: int = 0,
) -> pd.DataFrame:
    if benchmarks is None:
        benchmarks = DM_BENCHMARKS

    used_cols = [reference_model, *benchmarks, "y"]
    df = forecast_df[used_cols].copy()
    pred = df[[c for c in df.columns if c != "y"]]
    yv = df["y"].to_numpy().reshape(-1, 1)

    base_valid = np.isfinite(yv).ravel() & np.isfinite(pred.values).all(axis=1)
    y_base = yv[base_valid]
    pred_base = pred.iloc[base_valid].reset_index(drop=True)

    sigma = df["y"].shift(h).rolling(vol_window, min_periods=vol_window // 2).std()
    q_hi = sigma.rolling(vol_window, min_periods=vol_window // 3).quantile(quantile)
    q_lo = sigma.rolling(vol_window, min_periods=vol_window // 3).quantile(
        1.0 - quantile
    )
    high_vol = (sigma >= q_hi).fillna(False).to_numpy()[base_valid].astype(bool)
    low_vol = (sigma <= q_lo).fillna(False).to_numpy()[base_valid].astype(bool)

    rows = {"all": {}, "high_vol": {}, "low_vol": {}}
    for benchmark in benchmarks:
        loss = helpers.se(y_base, pred_base[[reference_model, benchmark]])
        loss_np = loss.values if hasattr(loss, "values") else np.asarray(loss)
        rows["all"][benchmark] = dm_signed_t(loss_np, bw=bw)
        rows["high_vol"][benchmark] = (
            dm_signed_t(loss_np[high_vol], bw=bw) if high_vol.sum() >= 3 else np.nan
        )
        rows["low_vol"][benchmark] = (
            dm_signed_t(loss_np[low_vol], bw=bw) if low_vol.sum() >= 3 else np.nan
        )

    dm_table = pd.DataFrame(rows).T[benchmarks]
    dm_table.index.name = "Volatility regime"
    reference_name = DISPLAY_NAMES.get(reference_model, reference_model)
    dm_table.columns.name = f"Benchmark model (reference: {reference_name})"
    return dm_table


def latex_dm_rows(
    dm_table: pd.DataFrame,
    include_header: bool = False,
    reference_model_label: str | None = None,
) -> str:
    lines = []
    if reference_model_label is not None:
        lines.append(f"% Reference model: {reference_model_label}")
    if include_header:
        header = "Regime & " + " & ".join(str(col) for col in dm_table.columns)
        lines.append(f"{header} \\\\ \\hline")
    for idx, row in dm_table.iterrows():
        values = " & ".join(f"{value:.3f}" for value in row)
        lines.append(f"{idx} & {values} \\\\ \\hline")
    return "\n".join(lines)


def run_diagnostics(
    selected_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 100,
    model_dir_overrides: dict[str, Path] | None = None,
) -> dict[str, object]:
    selected_dir = selected_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_models = [REFERENCE_MODEL, *DM_BENCHMARKS]
    forecast_df = load_forecast_matrix(
        selected_dir, all_models, seed, model_dir_overrides=model_dir_overrides
    )

    figure_paths = []
    diagnostic_paths = []
    for model in FIGURE_MODELS:
        calibrated = rolling_calibrate_df(
            forecast_df[[model, "y"]], window=252, step=60, intercept=True
        )
        figure_paths.append(plot_pred_vs_actual(calibrated, model, output_dir))
        figure_paths.append(plot_standardized_pred_vs_actual(calibrated, model, output_dir))
        figure_paths.append(plot_calibration(calibrated, model, output_dir))
        diagnostic_paths.extend(save_calibration_diagnostics(calibrated, model, output_dir))

    # Main paper table: compare the same winsorized forecast signal that enters
    # the trading rules.
    dm_forecast_df = winsorize_forecast_columns(forecast_df, all_models)
    dm_table = compute_dm_table(dm_forecast_df)

    dm_table.to_csv(output_dir / "ETF_dm_regimes_sp500.csv")
    (output_dir / "ETF_dm_regimes_sp500_latex_rows.txt").write_text(
        latex_dm_rows(
            dm_table.round(3),
            include_header=True,
            reference_model_label=(
                f"{DISPLAY_NAMES[REFERENCE_MODEL]} "
                f"(winsorized forecasts, window={DM_WINSOR_WINDOW})"
            ),
        )
        + "\n"
    )

    return {
        "forecast_df": forecast_df,
        "figure_paths": figure_paths,
        "diagnostic_paths": diagnostic_paths,
        "dm_table": dm_table,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce ETF appendix figures and DM-regime table."
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_dir = resolve_selected_dir(args.selected_dir)
    out = run_diagnostics(
        selected_dir=selected_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Wrote ETF diagnostics to: {out['output_dir']}")
    print(out["dm_table"].round(3))


if __name__ == "__main__":
    main()
