from __future__ import annotations

import argparse

from reproduce_simulation_common import (
    FIG_OUT,
    TABLE_OUT,
    add_common_log_args,
    ensure_output_dirs,
    plot_lines,
    read_model_tests,
    resolve_run_dir,
    write_table_and_figure_notice,
)


MORE_NONLINEAR_MODEL_MAP = {
    "oracleNNOpt": "oracleNN",
    "fAR-NNOpt": "FAR-NN",
    "vanillaNNOpt": "vanillaNN",
    "autoencoderOpt": "autoencoder",
    "pCA_NN_PCA_ADDOpt": "PCA_NN_PCA_ADD",
    "sPCA_NN_SPCA_ADDOpt": "SPCA_NN_SPCA_ADD",
    "nN_SPCA_NNOpt": "NN_SPCA_NN",
}

MORE_NONLINEAR_ORDER = [
    "vanillaNN",
    "autoencoder",
    "FAR-NN",
    "oracleNN",
    "NN_SPCA_NN",
    "PCA_NN_PCA_ADD",
    "SPCA_NN_SPCA_ADD",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce simulation more-nonlinear Figures 13--14 from saved logs.")
    add_common_log_args(
        parser,
        pattern="*Simluation_AllDim_nonlinear*simulationtrial_*",
        run_help="More-nonlinear simulation run directory. If omitted, the latest matching run is used.",
    )
    args = parser.parse_args()
    ensure_output_dirs()
    run_dir = resolve_run_dir(args.log_root, args.run_dir, args.auto_pattern)
    print(f"Using more-nonlinear simulation run: {run_dir}")

    results = read_model_tests(
        run_dir,
        MORE_NONLINEAR_MODEL_MAP,
        row_filter=lambda row: int(row["factor_id"]) == 1 and int(row["hcm_id"]) == 2,
        extra_fields=lambda row: {"b_f": int(row["b_f"])},
    )
    long_path = TABLE_OUT / "simulation_more_nonlinear_long_results.csv"
    results.to_csv(long_path, index=False)

    output_paths = [long_path]
    for b_f, fig_name in [(2, "ds12_table_bf2.png"), (3, "ds12_table_bf3.png")]:
        subset = results[results["b_f"] == b_f]
        if subset.empty:
            raise ValueError(f"No more-nonlinear rows found for b_f={b_f}")
        mean_tbl = (
            subset.groupby(["p", "model"])["test_mse"]
            .mean()
            .unstack()
            .reindex(columns=[m for m in MORE_NONLINEAR_ORDER if m in subset["model"].unique()])
            .sort_index()
        )
        count_tbl = (
            subset.groupby(["p", "model"])["seed"]
            .nunique()
            .unstack()
            .reindex(columns=mean_tbl.columns)
            .sort_index()
        )
        mean_path = TABLE_OUT / f"simulation_ds21_bf{b_f}_test_mse_by_p.csv"
        count_path = TABLE_OUT / f"simulation_ds21_bf{b_f}_n_seeds_by_p.csv"
        fig_path = FIG_OUT / fig_name
        mean_tbl.to_csv(mean_path)
        count_tbl.to_csv(count_path)
        plot_lines(
            mean_tbl.loc[50:3000],
            list(mean_tbl.columns),
            title=rf"DS21 test MSE vs dimension ($b_f={b_f}$)",
            output_path=fig_path,
        )
        output_paths.extend([mean_path, count_path, fig_path])

    write_table_and_figure_notice(output_paths)


if __name__ == "__main__":
    main()
