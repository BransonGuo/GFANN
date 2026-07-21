from __future__ import annotations

import argparse

import pandas as pd

from reproduce_simulation_common import (
    FIG_OUT,
    MAIN_MODEL_MAP,
    TABLE_OUT,
    add_common_log_args,
    aggregate_mean_and_counts,
    ensure_output_dirs,
    plot_lines,
    read_model_tests,
    resolve_run_dir,
    write_table_and_figure_notice,
)


FULL_DS11 = [
    "pcr", "arp", "di", "lasso", "pls", "vanillaNN", "autoencoder",
    "FAR-NN", "oracleNN", "SPCA_NN", "PCA_NN_PCA_ADD", "SPCA_NN_SPCA_ADD",
]
ZOOM_DS11 = ["FAR-NN", "oracleNN", "SPCA_NN", "PCA_NN_PCA_ADD", "SPCA_NN_SPCA_ADD"]
FULL_DS21 = [
    "pcr", "arp", "di", "lasso", "pls", "vanillaNN", "autoencoder",
    "FAR-NN", "oracleNN", "NN_SPCA_NN", "PCA_NN_PCA_ADD", "SPCA_NN_SPCA_ADD",
]
ZOOM_DS21 = ["FAR-NN", "oracleNN", "NN_SPCA_NN", "PCA_NN_PCA_ADD", "SPCA_NN_SPCA_ADD"]

TABLE2_SPECS = {
    "DS11": [
        ("SPCA_NN_SPCA_ADD", "1.2e4"),
        ("PCA_NN_PCA_ADD", "1.2e4"),
        ("oracleNN", "5.5e4"),
        ("FAR-NN", "2.5e5"),
        ("SPCA_NN", "1.2e5"),
        ("autoencoder", "1.5e5"),
        ("vanillaNN", "2.5e5"),
        ("pls", "2.5e3"),
        ("di", "2.5e3"),
        ("lasso", "5e2"),
        ("arp", "1"),
        ("pcr", "2.5e3"),
    ],
    "DS21": [
        ("PCA_NN_PCA_ADD", "1.2e4"),
        ("SPCA_NN_SPCA_ADD", "1.5e4"),
        ("oracleNN", "6.7e4"),
        ("FAR-NN", "2.5e5"),
        ("NN_SPCA_NN", "3.5e5"),
        ("autoencoder", "5.0e4"),
        ("vanillaNN", "2.8e5"),
        ("pls", "2.5e3"),
        ("di", "2.5e3"),
        ("lasso", "5e2"),
        ("pcr", "2.5e3"),
        ("arp", "1"),
    ],
}


def parse_seed_list(seed_text: str | None) -> list[int] | None:
    if not seed_text:
        return None
    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def build_table2(results: pd.DataFrame) -> pd.DataFrame:
    p500 = results[results["p"] == 500]
    rows = []
    for obs_id, dataset in enumerate(["DS11", "DS21"], start=1):
        for model, params in TABLE2_SPECS[dataset]:
            vals = p500[(p500["dataset"] == dataset) & (p500["model"] == model)]["test_mse"]
            if vals.empty:
                continue
            rows.append({
                "obs_id": obs_id,
                "target_id": 1,
                "dataset": dataset,
                "model": model,
                "trainable_params": params,
                "test_mse": vals.mean(),
                "std_test_mse": vals.std(ddof=1),
                "n_seeds": vals.count(),
                "test_mse_3dp": round(vals.mean(), 3),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce simulation Table 2 and Figures 7--8 from saved logs.")
    add_common_log_args(
        parser,
        pattern=(
            "*Simluation_AllDim_main*simulationtrial_*",
            "*Simluation_AllDimsimulationtrial_*",
        ),
        run_help="Main simulation run directory. If omitted, the latest matching run is used.",
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
    print(f"Using main simulation run: {run_dir}")

    results = read_model_tests(run_dir, MAIN_MODEL_MAP)
    seeds = parse_seed_list(args.seeds)
    if seeds is not None:
        results = results[results["seed"].isin(seeds)].copy()
        missing = sorted(set(seeds) - set(results["seed"].unique()))
        if missing:
            raise ValueError(f"Requested seeds not found in results: {missing}")
        print(f"Using seeds: {seeds}")
    long_path = TABLE_OUT / "simulation_main_long_results.csv"
    results.to_csv(long_path, index=False)

    mean, counts = aggregate_mean_and_counts(results, ["dataset", "p"])
    output_paths = [long_path]
    for dataset in ["DS11", "DS21"]:
        mean_tbl = mean.loc[dataset]
        count_tbl = counts.loc[dataset]
        mean_path = TABLE_OUT / f"simulation_{dataset}_test_mse_by_p.csv"
        count_path = TABLE_OUT / f"simulation_{dataset}_n_seeds_by_p.csv"
        mean_tbl.to_csv(mean_path)
        count_tbl.to_csv(count_path)
        output_paths.extend([mean_path, count_path])

    table2 = build_table2(results)
    table2_csv = TABLE_OUT / "simulation_table_test_mse_combined.csv"
    table2_tex = TABLE_OUT / "simulation_table_test_mse_combined.tex"
    table2.to_csv(table2_csv, index=False)
    table2.to_latex(table2_tex, index=False, float_format="%.3f")
    output_paths.extend([table2_csv, table2_tex])

    ds11 = mean.loc["DS11"]
    ds21 = mean.loc["DS21"]
    figs = [
        (ds11, FULL_DS11, "DS11 test MSE vs dimension", "ds02_table1.png"),
        (ds11, ZOOM_DS11, "DS11 test MSE vs dimension, zoomed", "ds02_table2.png"),
        (ds21, FULL_DS21, "DS21 test MSE vs dimension", "ds12_table1.png"),
        (ds21, ZOOM_DS21, "DS21 test MSE vs dimension, zoomed", "ds12_table2.png"),
    ]
    for table, cols, title, filename in figs:
        fig_path = FIG_OUT / filename
        plot_lines(table.loc[50:3000], cols, title=title, output_path=fig_path)
        output_paths.append(fig_path)

    write_table_and_figure_notice(output_paths)


if __name__ == "__main__":
    main()
