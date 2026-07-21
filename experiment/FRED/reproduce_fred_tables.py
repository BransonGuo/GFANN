"""Reproduce FRED-MD paper tables from saved experiment summaries.

This script currently reproduces Table ``FRED_rank_metrics`` from the saved
``summary_file.csv`` files. It mirrors the notebook logic:

    df_score_all.rank(axis=1).mean()

where rows are FRED target variables and columns are model test scores. The
score is ``-R^2_OOS``, so lower values are better and rank 1 is best.

By default, the script uses the latest timestamped FRED run under
``logs/FRED`` that contains ``fred_idx*/summary_file.csv`` files.
You can also pass ``--run-dir`` explicitly.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FRED_LOG_ROOT = REPO_ROOT / "logs" / "FRED"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "table_outputs"

TARGET_COUNT = 127

TABLE_MODEL_COLUMNS = [
    "gFANNOpt_test_score",
    "lasso_test_score",
    "fan_fast_test_score",
    "pls_test_score",
    "di_test_score",
    "arp_test_score",
    "vanillaNNOpt_test_score",
    "pcr_test_score",
]

DISPLAY_NAMES = {
    "gFANNOpt_test_score": "GFANN",
    "pCAA_NNOpt_var_test_score": "GFANN",
    "lasso_test_score": "lasso",
    "fan_fast_test_score": "FAST-NN",
    "pls_test_score": "pls",
    "di_test_score": "di",
    "arp_test_score": "arp",
    "vanillaNNOpt_test_score": "vanillaNN",
    "pcr_test_score": "pcr",
}

PARAM_COUNTS_LATEX = {
    "GFANN": r"$2\times10^{3}$",
    "lasso": r"$1.3\times10^{2}$",
    "FAST-NN": r"$7\times10^{3}$",
    "pls": r"$6\times10^{2}$",
    "di": r"$5\times10^{2}$",
    "arp": r"$2\times10^{0}$",
    "vanillaNN": r"$4.7\times10^{3}$",
    "pcr": r"$5\times10^{2}$",
}

LONGTABLE_COLUMNS = [
    "pCAA_NNOpt_var_test_score",
    "lasso_test_score",
    "fan_fast_test_score",
    "pls_test_score",
    "di_test_score",
    "arp_test_score",
    "vanillaNNOpt_test_score",
    "pcr_test_score",
]

LONGTABLE_HEADERS = (
    r"\textbf{target} & \makecell{\textbf{GFA-}\\\textbf{NN}} & "
    r"\textbf{lasso} & \makecell{\textbf{FAST-}\\\textbf{NN}} & "
    r"\textbf{pls} & \textbf{di} & \textbf{arp} & "
    r"\makecell{\textbf{vanilla-}\\\textbf{NN}} & \textbf{pcr} \\"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute the FRED average-rank table from saved summaries."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Timestamped FRED run containing all paper models. If omitted, "
            "the latest run under logs/FRED is used."
        ),
    )
    parser.add_argument(
        "--gfann-dir",
        type=Path,
        default=None,
        help=(
            "Optional separate directory containing fred_idx*/summary_file.csv "
            "for GFANN. If one separate model directory is provided, all four "
            "separate directory arguments must be provided."
        ),
    )
    parser.add_argument(
        "--fast-dir",
        type=Path,
        default=None,
        help="Optional separate directory containing FAST-NN summaries.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Optional separate directory containing benchmark model summaries.",
    )
    parser.add_argument(
        "--lasso-pcr-dir",
        type=Path,
        default=None,
        help="Optional separate directory containing Lasso and PCR summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where reproduced CSV/LaTeX files are written.",
    )
    parser.add_argument(
        "--rank-method",
        default="average",
        choices=["average", "min", "max", "first", "dense"],
        help="Pandas rank tie method. The notebook's rank(axis=1).mean() uses average.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def find_latest_fred_run(log_root: Path = FRED_LOG_ROOT) -> Path:
    if not log_root.exists():
        raise FileNotFoundError(
            f"FRED log root does not exist: {log_root}. "
            "Run experiment/FRED/run_scripts_FRED.py first, "
            "or pass --run-dir."
        )
    candidates = [
        p
        for p in log_root.iterdir()
        if p.is_dir() and any(p.glob("fred_idx*/summary_file.csv"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No FRED run with fred_idx*/summary_file.csv found under {log_root}. "
            "Run experiment/FRED/run_scripts_FRED.py first, "
            "or pass --run-dir."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


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
    if any(path is not None for path in separate_dirs):
        if not all(path is not None for path in separate_dirs):
            raise ValueError(
                "Pass --run-dir, or provide all of --gfann-dir, --fast-dir, "
                "--benchmark-dir, and --lasso-pcr-dir."
            )
        return tuple(resolve_path(path) for path in separate_dirs)  # type: ignore[arg-type]

    run_dir = find_latest_fred_run()
    return run_dir, run_dir, run_dir, run_dir


def read_summary_scores(folder: Path) -> pd.DataFrame:
    rows = []
    for child in sorted(folder.iterdir()):
        summary_path = child / "summary_file.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        score_cols = [
            col
            for col in df.columns
            if re.search(r"test|valid", col) or col in {"seed", "fred_idx"}
        ]
        rows.append(df.loc[:, score_cols])

    if not rows:
        raise FileNotFoundError(f"No summary_file.csv files found under {folder}")

    out = pd.concat(rows, ignore_index=True)
    out = out.set_index("fred_idx").sort_index()
    if out.index.nunique() != TARGET_COUNT:
        raise ValueError(
            f"Expected {TARGET_COUNT} FRED targets in {folder}, "
            f"found {out.index.nunique()}."
        )
    return out


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of these columns were found: {candidates}")


def build_score_matrix(
    gfann_dir: Path,
    fast_dir: Path,
    benchmark_dir: Path,
    lasso_pcr_dir: Path,
) -> pd.DataFrame:
    gfann = read_summary_scores(gfann_dir)
    fast = read_summary_scores(fast_dir)
    benchmark = read_summary_scores(benchmark_dir)
    lasso_pcr = read_summary_scores(lasso_pcr_dir)

    gfann_col = first_existing_column(
        gfann, ["gFANNOpt_test_score", "pCAA_NNOpt_var_test_score"]
    )

    score_df = pd.concat(
        [
            gfann[[gfann_col]].rename(
                columns={gfann_col: "gFANNOpt_test_score"}
            ),
            fast[["fan_fast_test_score"]],
            benchmark[
                [
                    "vanillaNNOpt_test_score",
                    "pls_test_score",
                    "di_test_score",
                    "arp_test_score",
                ]
            ],
            lasso_pcr[["lasso_test_score", "pcr_test_score"]],
        ],
        axis=1,
    ).sort_index()

    missing = score_df.isna().sum().sum()
    if missing:
        raise ValueError(f"Combined score matrix contains {missing} missing values.")
    return score_df[TABLE_MODEL_COLUMNS]


def compute_rank_table(score_df: pd.DataFrame, rank_method: str) -> pd.DataFrame:
    avg_rank = score_df.rank(axis=1, method=rank_method, ascending=True).mean()
    out = (
        avg_rank.rename("avg_rank")
        .rename(index=DISPLAY_NAMES)
        .sort_values()
        .reset_index()
        .rename(columns={"index": "model"})
    )
    out["n_targets"] = TARGET_COUNT
    out["rank_method"] = rank_method
    out["score_definition"] = "-R2_OOS"
    return out[["model", "avg_rank", "n_targets", "rank_method", "score_definition"]]


def fred_target_names() -> list[str]:
    modern_path = REPO_ROOT / "data" / "FRED-MD" / "modern.csv"
    modern = pd.read_csv(modern_path, nrows=1)
    return list(modern.columns[1:])


def build_longtable(score_df: pd.DataFrame) -> pd.DataFrame:
    target_names = fred_target_names()
    out = score_df.copy()
    out["fred_name"] = [target_names[int(idx)] for idx in out.index]
    out["diff"] = out["fan_fast_test_score"] - out["gFANNOpt_test_score"]
    out = out.sort_values("diff", ascending=False)
    out = out.rename(columns={"gFANNOpt_test_score": "pCAA_NNOpt_var_test_score"})
    return out[["fred_name", *LONGTABLE_COLUMNS]].round(3)


def latex_longtable_rows(score_table: pd.DataFrame) -> str:
    lines = [
        r"% Requires: \usepackage{longtable,booktabs,makecell}",
        r"\begin{longtable}{lrrrrrrrr}",
        r"\toprule",
        LONGTABLE_HEADERS,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        LONGTABLE_HEADERS,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{9}{r}{\emph{Continued on next page}}\\",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in score_table.iterrows():
        values = " & ".join(f"{row[col]:.3f}" for col in LONGTABLE_COLUMNS)
        lines.append(f"{row['fred_name']} & {values} " + r"\\")
    lines.extend(
        [
            r"\caption{Comparison of out-of-sample model performance for all targets; the metric is negative $R^2_{\text{OOS}}$ (lower is better).}",
            r"\label{table:FRED_model_performance}\\",
            r"\end{longtable}",
        ]
    )
    return "\n".join(lines)


def latex_rows(rank_table: pd.DataFrame) -> str:
    models = rank_table["model"].tolist()
    ranks = rank_table["avg_rank"].tolist()
    params = [PARAM_COUNTS_LATEX[m] for m in models]

    header = r"\textbf{Metric} & " + " & ".join(rf"\textbf{{{m}}}" for m in models)
    rank_row = r"\textbf{Avg rank} & " + " & ".join(f"{value:.3f}" for value in ranks)
    param_row = r"\textbf{\# params} & " + " & ".join(params)
    return "\n".join(
        [
            header + r" \\ \hline",
            rank_row + r" \\ \hline",
            param_row + r" \\ \hline",
        ]
    )


def write_outputs(score_df: pd.DataFrame, rank_table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    longtable = build_longtable(score_df)
    score_df.to_csv(output_dir / "FRED_score_all.csv")
    rank_table.to_csv(output_dir / "FRED_rank_metrics.csv", index=False)
    longtable.to_csv(output_dir / "FRED_model_performance_longtable_source.csv", index=False)
    (output_dir / "fred_performance_longtable.tex").write_text(
        latex_longtable_rows(longtable) + "\n"
    )
    (output_dir / "FRED_rank_metrics_latex_rows.txt").write_text(
        latex_rows(rank_table) + "\n"
    )


def main() -> None:
    args = parse_args()
    gfann_dir, fast_dir, benchmark_dir, lasso_pcr_dir = resolve_input_dirs(args)

    score_df = build_score_matrix(
        gfann_dir,
        fast_dir,
        benchmark_dir,
        lasso_pcr_dir,
    )
    rank_table = compute_rank_table(score_df, args.rank_method)
    write_outputs(score_df, rank_table, args.output_dir)

    print(f"Wrote FRED rank-table outputs to: {args.output_dir}")
    print(rank_table.round({"avg_rank": 3}).to_string(index=False))


if __name__ == "__main__":
    main()
