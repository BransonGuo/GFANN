# FRED Experiment Reproduction

This folder contains the script-based reproduction workflow for the FRED-MD experiment. The reviewer-facing workflow uses one run script and one experiment file:

- `run_scripts_FRED.py`: launches the FRED experiment over all 127 FRED-MD targets.
- `exp_FRED.py`: fits the requested model list for one target and writes the target-level outputs.

No notebook is required to reproduce the reported FRED tables or figures.

Run commands from the repository root:

```bash
cd <repo-root>
```

## 1. Run The FRED Experiment

Run:

```bash
python experiment/FRED/run_scripts_FRED.py
```

By default, this script runs the paper model set:

```text
fan_fast, vanillaNNOpt, lasso, pcr, pls, di, arp, gFANNOpt
```

It writes one timestamped folder under:

```text
logs/FRED/
```

Each target has a subfolder named like `fred_idx0_trial150`, containing:

- `summary_file.csv`: validation/test losses and scores.
- `df_pred.csv`: saved test-set predictions used for diagnostics.
- `df_param.csv`: selected model parameters where applicable.

For example, a fresh run may create:

```text
logs/FRED/<TIMESTAMP>FRED_ALL_freeze100_16FRED/
```

Use this folder as `<FRED_RUN>` in the commands below.

## 2. Reproduce FRED Tables

After running the experiment, regenerate the FRED table sources with:

```bash
python experiment/FRED/reproduce_fred_tables.py \
  --run-dir logs/FRED/<FRED_RUN> \
  --output-dir experiment/FRED/table_outputs
```

This writes:

```text
experiment/FRED/table_outputs/FRED_score_all.csv
experiment/FRED/table_outputs/FRED_rank_metrics.csv
experiment/FRED/table_outputs/FRED_rank_metrics_latex_rows.txt
experiment/FRED/table_outputs/FRED_model_performance_longtable_source.csv
experiment/FRED/table_outputs/fred_performance_longtable.tex
```

These files reproduce:

- `\label{table:FRED_rank_metrics}`: average rank across 127 FRED targets.
- `\label{table:FRED_model_performance}`: full target-by-target FRED performance table.

The table metric is negative out-of-sample \(R^2\), so lower values and lower ranks are better.

## 3. Reproduce FRED Diagnostic Figures And DM Table

Regenerate the FRED diagnostic figures and volatility-regime Diebold-Mariano table with:

```bash
python experiment/FRED/reproduce_fred_diagnostics.py \
  --run-dir logs/FRED/<FRED_RUN> \
  --output-dir experiment/FRED/diagnostic_outputs
```

By default, this command generates only the FRED diagnostic figures used in the
paper:

```text
idx0  = RPI     (Figure 19)
idx91 = T10YFFM (fig:fred-pva-more)
```

This writes:

```text
experiment/FRED/diagnostic_outputs/FRED_dm_regimes.csv
experiment/FRED/diagnostic_outputs/FRED_dm_regimes_latex_rows.txt
experiment/FRED/diagnostic_outputs/FRED_diagnostic_manifest.csv
experiment/FRED/diagnostic_outputs/FRED_pred_vs_actual_*.png
experiment/FRED/diagnostic_outputs/FRED_bin_calibration_*.png
```

These reproduce:

- `\label{fig:FRED_pred_vs_actual}`: predicted-vs-actual diagnostics for RPI (`idx0`).
- `\label{fig:FRED_bin_calibration}`: prediction-bin calibration curves for RPI (`idx0`).
- `\label{fig:fred-pva-more}`: additional FRED diagnostic for T10YFFM (`idx91`).
- `\label{tab:dm_regimes}`: volatility-regime Diebold-Mariano table.

The manifest file maps each generated PNG to the target index, FRED variable
name, model, diagnostic type, and paper label/panel.

To generate plots for all 127 targets, use:

```bash
python experiment/FRED/reproduce_fred_diagnostics.py \
  --run-dir logs/FRED/<FRED_RUN> \
  --targets $(seq 0 126) \
  --output-dir experiment/FRED/diagnostic_outputs_all
```
