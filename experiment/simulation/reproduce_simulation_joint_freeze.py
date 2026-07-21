from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reproduce_simulation_common import (
    DATASET_MAP,
    DEFAULT_LOG_ROOT,
    FIG_OUT,
    REPO_ROOT,
    TABLE_OUT,
    ensure_output_dirs,
    format_patterns,
    matches_any_pattern,
    plot_lines,
    read_model_tests,
    resolve_run_dir,
    write_table_and_figure_notice,
)


SPCA_MAP = {"sPCA_NNOpt": "SPCA_NN"}


def latest_matching(log_root: Path, pattern: str | list[str] | tuple[str, ...], *, exclude: Path | None = None) -> Path:
    candidates = [p for p in log_root.iterdir() if p.is_dir() and matches_any_pattern(p.name, pattern)]
    if exclude is not None:
        candidates = [p for p in candidates if p.resolve() != exclude.resolve()]
    if not candidates:
        raise FileNotFoundError(f"No run directory matching {format_patterns(pattern)} under {log_root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def read_spca(run_dir: Path, setting: str) -> pd.DataFrame:
    out = read_model_tests(run_dir, SPCA_MAP, dataset_map=DATASET_MAP)
    out["setting"] = setting
    return out


def resolve_mode_dir(combined_dir: Path, mode: str) -> Path:
    direct = combined_dir / mode
    if not direct.exists():
        raise FileNotFoundError(f"Missing {mode!r} directory under combined run dir: {combined_dir}")

    # Current joint-train runner writes mode/trial_*/p_*/seed*/summary_file.csv.
    trial_dirs = sorted(p for p in direct.glob("trial_*") if p.is_dir())
    if trial_dirs:
        if len(trial_dirs) > 1:
            raise ValueError(f"Multiple trial directories found under {direct}: {trial_dirs}")
        return trial_dirs[0]

    # Clean archives may use mode/p_*/seed*/summary_file.csv directly.
    return direct


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce simulation Figure 12: frozen SPCA_NN vs joint-train SPCA_NN.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--combined-run-dir",
        type=Path,
        default=None,
        help=(
            "Single run directory containing joint/ and freeze/ subdirectories. "
            "If supplied, --freeze-run-dir and --joint-run-dir are ignored."
        ),
    )
    parser.add_argument(
        "--freeze-run-dir",
        type=Path,
        default=None,
        help="Main simulation run directory for frozen SPCA_NN. Defaults to the latest matching main run.",
    )
    parser.add_argument(
        "--joint-run-dir",
        type=Path,
        default=None,
        help="Joint-train simulation run directory. Defaults to the latest matching joint run.",
    )
    args = parser.parse_args()
    ensure_output_dirs()

    if args.combined_run_dir is not None:
        combined_dir = args.combined_run_dir
        if not combined_dir.is_absolute():
            combined_dir = REPO_ROOT / combined_dir
        if not combined_dir.exists():
            raise FileNotFoundError(f"Combined run directory does not exist: {combined_dir}")
        freeze_dir = resolve_mode_dir(combined_dir, "freeze")
        joint_dir = resolve_mode_dir(combined_dir, "joint")
    else:
        freeze_dir = resolve_run_dir(
            args.log_root,
            args.freeze_run_dir,
            (
                "*Simluation_AllDim_main*simulationtrial_*",
                "*Simluation_AllDimsimulationtrial_*",
            ),
        )
        if args.joint_run_dir is None:
            joint_dir = latest_matching(args.log_root, "*Simluation_AllDim_joint*simulationtrial_*", exclude=freeze_dir)
        else:
            joint_dir = args.joint_run_dir
            if not joint_dir.is_absolute():
                joint_dir = REPO_ROOT / joint_dir
            if not joint_dir.exists():
                raise FileNotFoundError(f"Joint run directory does not exist: {joint_dir}")

    print(f"Using frozen SPCA_NN run: {freeze_dir}")
    print(f"Using joint-train SPCA_NN run: {joint_dir}")

    frozen = read_spca(freeze_dir, "Frozen-SPCA_NN")
    joint = read_spca(joint_dir, "Joint-SPCA_NN")
    results = pd.concat([frozen, joint], ignore_index=True)
    long_path = TABLE_OUT / "simulation_joint_train_vs_freeze_spca_long_results.csv"
    results.to_csv(long_path, index=False)

    for dataset in ["DS11", "DS21"]:
        has_joint = not joint[joint["dataset"] == dataset].empty
        if not has_joint:
            raise ValueError(
                f"No joint-train SPCA_NN results found for {dataset}. "
                "Wait for run_scripts_simulation_joint_train.py to finish."
            )

    counts = (
        results.groupby(["dataset", "p", "setting"])["seed"]
        .nunique()
        .unstack("setting")
        .sort_index()
    )
    count_path = TABLE_OUT / "simulation_joint_train_vs_freeze_spca_n_seeds_by_p.csv"
    counts.to_csv(count_path)

    output_paths = [long_path, count_path]
    for dataset, fig_name in [("DS11", "ds02_joint_SPCA_NN.png"), ("DS21", "ds12_joint_SPCA_NN.png")]:
        subset = results[results["dataset"] == dataset]
        mean_tbl = (
            subset.groupby(["p", "setting"])["test_mse"]
            .mean()
            .unstack("setting")
            .reindex(columns=["Frozen-SPCA_NN", "Joint-SPCA_NN"])
            .sort_index()
        )
        mean_path = TABLE_OUT / f"simulation_{dataset}_joint_train_vs_freeze_spca.csv"
        fig_path = FIG_OUT / fig_name
        mean_tbl.to_csv(mean_path)
        plot_lines(
            mean_tbl.loc[50:3000],
            ["Frozen-SPCA_NN", "Joint-SPCA_NN"],
            title=f"{dataset} SPCA_NN: frozen vs joint train",
            output_path=fig_path,
        )
        output_paths.extend([mean_path, fig_path])

    write_table_and_figure_notice(output_paths)


if __name__ == "__main__":
    main()
