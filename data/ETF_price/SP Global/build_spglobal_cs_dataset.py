from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "data/ETF_price/SP Global/AllETFSimpleRet 10yrs.csv"
)
DEFAULT_OUTPUT = Path(
    "data/ETF_price/SP Global/"
    "cs_SimpleRet1-5-10-20-126_yLags1-20_SP500 as Y.csv.gz"
)
ROLLING_WINDOWS = [5, 10, 20, 126]
DEFAULT_Y_LAG_MAX = 20
N_STOCK = 11


def parse_y_lags(raw_value: str) -> list[int]:
    y_lags = []
    for part in raw_value.split(","):
        token = part.strip()
        if not token:
            continue
        lag = int(token)
        if lag <= 0:
            raise ValueError(f"y lag must be positive, got {lag}")
        if lag not in y_lags:
            y_lags.append(lag)
    if not y_lags:
        raise ValueError("At least one y lag must be provided.")
    return y_lags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the S&P Global sector-index cross-sectional dataset, "
            "with optional lagged y terms appended to the feature matrix."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to AllETFSimpleRet 10yrs.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the gzip CSV.",
    )
    parser.add_argument(
        "--y-lag-max",
        type=int,
        default=DEFAULT_Y_LAG_MAX,
        help="Append y_lag1 ... y_lagN to the feature matrix.",
    )
    parser.add_argument(
        "--y-lags",
        type=str,
        default="",
        help="Explicit comma-separated lag list such as '1,5,10,20,126'.",
    )
    return parser.parse_args()


def build_cs_dataset(df_all_ret: pd.DataFrame, y_lags: list[int]) -> pd.DataFrame:
    feature_source = df_all_ret.iloc[:, :N_STOCK].copy()

    feature_frames = [feature_source.add_suffix("_ret1")]
    for window in ROLLING_WINDOWS:
        rolled = feature_source.rolling(window).sum()
        rolled.columns = [f"{col}_ret{window}" for col in feature_source.columns]
        feature_frames.append(rolled)

    if y_lags:
        target_base = df_all_ret["SP500"]
        for lag in y_lags:
            feature_frames.append(target_base.shift(lag - 1).rename(f"y_lag{lag}"))

    df_all_cs = pd.concat(feature_frames, axis=1)
    df_all_cs["y"] = df_all_ret["SP500"].shift(-1)
    df_all_cs.index.name = "date"
    return df_all_cs.dropna()


def main() -> None:
    args = parse_args()
    y_lags = parse_y_lags(args.y_lags) if args.y_lags else list(range(1, args.y_lag_max + 1))
    df_all_ret = pd.read_csv(args.input, index_col=0, parse_dates=True)
    df_all_cs = build_cs_dataset(df_all_ret, y_lags=y_lags)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_all_cs.to_csv(
        args.output,
        compression="gzip",
        index_label="date",
        date_format="%Y-%m-%d",
    )

    print(f"input_shape={df_all_ret.shape}")
    print(f"output_shape={df_all_cs.shape}")
    print(f"output_start={df_all_cs.index.min().date()}")
    print(f"output_end={df_all_cs.index.max().date()}")
    print(f"y_lags={y_lags}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
