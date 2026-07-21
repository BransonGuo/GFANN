from __future__ import annotations

import argparse

from reproduce_simulation_common import (
    FIG_OUT,
    TABLE_OUT,
    add_common_log_args,
    aggregate_mean_and_counts,
    ensure_output_dirs,
    plot_lines,
    read_model_tests,
    resolve_run_dir,
    write_table_and_figure_notice,
)


ABLA_MODEL_MAP = {
    "pCA_NN_PCAOpt": "PCA_NN_PCA",
    "pCA_NN_ADD_PCAOpt": "PCA_NN_ADD_PCA",
    "pCA_NN_PCA_ADDOpt": "PCA_NN_PCA_ADD",
    "sPCA_NN_SPCA_ADDOpt": "SPCA_NN_SPCA_ADD",
    "sPCA_NN_SPCAOpt": "SPCA_NN_SPCA",
    "sPCA_NN_ADD_SPCAOpt": "SPCA_NN_ADD_SPCA",
}

ABLA_ORDER = [
    "PCA_NN_PCA_ADD",
    "PCA_NN_ADD_PCA",
    "PCA_NN_PCA",
    "SPCA_NN_SPCA_ADD",
    "SPCA_NN_ADD_SPCA",
    "SPCA_NN_SPCA",
]


def parse_seed_list(seed_text: str | None) -> list[int] | None:
    if not seed_text:
        return None
    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce simulation ablation Figures 10--11 from saved logs.")
    add_common_log_args(
        parser,
        pattern="*Simluation_AllDim_ablation*simulationtrial_*",
        run_help="Ablation simulation run directory. If omitted, the latest matching run is used.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list to include, e.g. 100,105,...,145. If omitted, all seeds are used.",
    )
    args = parser.parse_args()
    ensure_output_dirs()
    run_dir = resolve_run_dir(args.log_root, args.run_dir, args.auto_pattern)
    print(f"Using ablation simulation run: {run_dir}")

    results = read_model_tests(run_dir, ABLA_MODEL_MAP)
    seeds = parse_seed_list(args.seeds)
    if seeds is not None:
        results = results[results["seed"].isin(seeds)].copy()
        missing = sorted(set(seeds) - set(results["seed"].unique()))
        if missing:
            raise ValueError(f"Requested seeds not found in results: {missing}")
        print(f"Using seeds: {seeds}")
    long_path = TABLE_OUT / "simulation_ablation_long_results.csv"
    results.to_csv(long_path, index=False)
    mean, counts = aggregate_mean_and_counts(results, ["dataset", "p"])

    output_paths = [long_path]
    for dataset, fig_name in [("DS11", "ablation1.png"), ("DS21", "ablation2.png")]:
        mean_tbl = mean.loc[dataset].reindex(columns=[m for m in ABLA_ORDER if m in mean.loc[dataset].columns])
        count_tbl = counts.loc[dataset].reindex(columns=mean_tbl.columns)
        mean_path = TABLE_OUT / f"simulation_ablation_{dataset}_test_mse_by_p.csv"
        count_path = TABLE_OUT / f"simulation_ablation_{dataset}_n_seeds_by_p.csv"
        fig_path = FIG_OUT / fig_name
        mean_tbl.to_csv(mean_path)
        count_tbl.to_csv(count_path)
        plot_lines(mean_tbl, list(mean_tbl.columns), title=f"{dataset} ablation: test MSE vs dimension", output_path=fig_path)
        output_paths.extend([mean_path, count_path, fig_path])

    write_table_and_figure_notice(output_paths)


if __name__ == "__main__":
    main()
