# Simulation Replication Scripts

This folder separates computational experiment reruns from paper table/figure regeneration.

## 1. Rerun Experiments

Run these scripts from the repository root. They write timestamped logs under `logs/Simulation/`.

```bash
python experiment/simulation/run_scripts_simulation.py
python experiment/simulation/run_scripts_simulation_ablation.py
python experiment/simulation/run_scripts_simulation_more_nonlinear.py
python experiment/simulation/run_scripts_simulation_joint_train.py
```

These runs are computationally expensive. The reviewer-facing defaults use the
10 paper seeds `100,105,...,145` for the simulation figures and tables.
The default reviewer-facing scripts use the paper model spaces. For the joint/freeze comparison,
`run_scripts_simulation_joint_train.py` writes both variants by default under a single timestamped
directory with `joint/` and `freeze/` subfolders; set `SIM_RUN_MODES=joint` or
`SIM_RUN_MODES=freeze` only if you intentionally want to run one side.
The more-nonlinear and joint/freeze scripts use early-stopping patience 30 by
default; override with `SIM_PATIENCE=<value>` if needed.

The main hard-PCA specifications use a one-shot internal PCA update at epoch 1
(`init_schedule_ori = [1]`). The dedicated Figures 2--4 diagnostic instead
uses its explicitly stated multi-epoch operating schedule.

## 2. Reproduce Figures 2--4: PCA Operating Schedule

Figures 2--4 are a dedicated diagnostic experiment rather than part of the
main Monte Carlo aggregation. The shared `PcaLayer` forwards the update flag to
the underlying PCA operation. The diagnostic therefore recomputes the second
PCA basis only at the stated update epochs, without using a separate layer
class.

Run the paper specification from the repository root:

```bash
PCA_SCHEDULE_RUN="logs/Simulation/figure2_3_pca_schedule"

python experiment/simulation/run_simulation_pca_schedule.py \
  --model pca_nn_pca \
  --n-trials 100 \
  --tune-condition scheduled \
  --tune-patience 100 \
  --seeds 100 \
  --factor-id 1 \
  --hcm-id 2 \
  --p 500 \
  --epochs 300 \
  --output-dir "$PCA_SCHEDULE_RUN"
```

Generate Figures 2 and 3 from the saved loss history:

```bash
python experiment/simulation/reproduce_simulation_pca_schedule.py \
  --input "$PCA_SCHEDULE_RUN/pca_schedule_loss_history.csv"
```

Generate Figure 4 using the same selected configuration, simulated sample,
initialization, and update schedule:

```bash
python experiment/simulation/run_pca_explained_variance.py \
  --config "$PCA_SCHEDULE_RUN/pca_schedule_config.json"
```

The principal outputs are:

```text
logs/Simulation/figure2_3_pca_schedule/train_valid_loss_no_schedule.PNG
logs/Simulation/figure2_3_pca_schedule/train_valid_loss_schedule.PNG
logs/Simulation/figure2_3_pca_schedule/figure4_explained_variance/pct_variance_explained_with_train_valid.PNG
```

If the separately supplied simulation intermediary archive has been extracted
under `logs/Simulation/reviewer_clean_results_package`, Figures 2--4 can be
redrawn without model retraining:

```bash
SIM_SAVED="logs/Simulation/reviewer_clean_results_package"
PCA_SAVED="$SIM_SAVED/pca_schedule_figures_2_4"

python experiment/simulation/reproduce_simulation_pca_schedule.py \
  --input "$PCA_SAVED/pca_schedule_loss_history.csv" \
  --output-dir "$PCA_SAVED/reproduced_figures"

python experiment/simulation/run_pca_explained_variance.py \
  --history "$PCA_SAVED/figure4_explained_variance/figure4_explained_variance_history.csv" \
  --retained-components 10 \
  --output-dir "$PCA_SAVED/reproduced_figures"
```

## 3. Reproduce Figure 9: Hyperparameter Importance

Figure 9 uses the same model implementation as the main simulation, but uses a
dedicated broader sensitivity search space so that architecture parameters
which are fixed in the main experiment can receive fANOVA importances. The
paper specification runs 50 TPE trials for each of 10 seeds and three models:

```bash
python experiment/simulation/run_scripts_simulation_hyperparameter_importance.py
```

The command prints the completed run root. Pass that single directory to:

```bash
FIGURE9_RUN="logs/Simulation/<FIGURE9_RUN_DIRECTORY>"

python experiment/simulation/reproduce_simulation_hyperparameter_importance.py \
  --run-root "$FIGURE9_RUN"
```

To inspect the exact commands and output structure without starting training:

```bash
python experiment/simulation/run_scripts_simulation_hyperparameter_importance.py \
  --seeds 100 \
  --run-name figure9_smoke_test \
  --dry-run
```

The full reproduction writes the three paper panels and their underlying CSV
files to `experiment/simulation/figure9_outputs/`.

The separately supplied simulation intermediary archive also contains the 30
saved Optuna studies underlying Figure 9. After extracting it under
`logs/Simulation/reviewer_clean_results_package`, regenerate the fANOVA CSVs
and all three panels without retraining:

```bash
SIM_SAVED="logs/Simulation/reviewer_clean_results_package"

python experiment/simulation/reproduce_simulation_hyperparameter_importance.py \
  --run-root "$SIM_SAVED/hyperparameter_importance_figure_9" \
  --output-dir "$SIM_SAVED/hyperparameter_importance_figure_9/reproduced_outputs"
```

## 4. Regenerate Main Simulation Tables And Figures

After the corresponding experiment logs exist, set the run directories created
by the run scripts. Explicit paths are recommended because automatic pattern
matching may select the wrong run if several similar timestamped directories
are present.

```bash
MAIN_RUN="logs/Simulation/<MAIN_SIMULATION_RUN>"
ABLA_RUN="logs/Simulation/<ABLATION_RUN>"
NONLINEAR_RUN="logs/Simulation/<MORE_NONLINEAR_RUN>"
JOINT_FREEZE_RUN="logs/Simulation/<JOINT_FREEZE_RUN>"
```

Then regenerate the tables and figures:

```bash
python experiment/simulation/reproduce_simulation_main.py \
  --run-dir "$MAIN_RUN"

python experiment/simulation/reproduce_simulation_ablation.py \
  --run-dir "$ABLA_RUN"

python experiment/simulation/reproduce_simulation_more_nonlinear.py \
  --run-dir "$NONLINEAR_RUN"

python experiment/simulation/reproduce_simulation_joint_freeze.py \
  --combined-run-dir "$JOINT_FREEZE_RUN"
```

If the frozen and jointly trained SPCA_NN results were produced in separate
directories, use:

```bash
python experiment/simulation/reproduce_simulation_joint_freeze.py \
  --freeze-run-dir "$FREEZE_RUN" \
  --joint-run-dir "$JOINT_RUN"
```

To restrict the main or ablation summaries to the first 10 paper seeds, pass:

```bash
--seeds 100,105,110,115,120,125,130,135,140,145
```

As a convenience, the following command runs all four reproduction scripts using
automatic latest-run discovery:

```bash
python experiment/simulation/reproduce_simulation_all.py
```

Use this convenience command only when there is a single clear run directory for
each simulation component. Otherwise, use the explicit commands above.

## Paper Mapping

| Paper item | Rerun script | Reproduction script | Main outputs |
|---|---|---|---|
| Figure 2 | `run_simulation_pca_schedule.py` | `reproduce_simulation_pca_schedule.py` | `train_valid_loss_no_schedule.PNG` |
| Figure 3 | `run_simulation_pca_schedule.py` | `reproduce_simulation_pca_schedule.py` | `train_valid_loss_schedule.PNG` |
| Figure 4 | `run_simulation_pca_schedule.py` | `run_pca_explained_variance.py` | `figure4_explained_variance/pct_variance_explained_with_train_valid.PNG` |
| Figure 9 | `run_scripts_simulation_hyperparameter_importance.py` | `reproduce_simulation_hyperparameter_importance.py` | `figure9_outputs/hyperpara_farnn.png`, `hyperpara_ori_pca_pca_add.png`, `hyperpara_pca_pca_add.png` |
| Table 2 | `run_scripts_simulation.py` | `reproduce_simulation_main.py` | `table_outputs/simulation_table_test_mse_combined.csv`, `.tex` |
| Figure 7 | `run_scripts_simulation.py` | `reproduce_simulation_main.py` | `figure_outputs/ds02_table1.png`, `figure_outputs/ds02_table2.png` |
| Figure 8 | `run_scripts_simulation.py` | `reproduce_simulation_main.py` | `figure_outputs/ds12_table1.png`, `figure_outputs/ds12_table2.png` |
| Figure 10 | `run_scripts_simulation_ablation.py` | `reproduce_simulation_ablation.py` | `figure_outputs/ablation1.png` |
| Figure 11 | `run_scripts_simulation_ablation.py` | `reproduce_simulation_ablation.py` | `figure_outputs/ablation2.png` |
| Figure 12 | `run_scripts_simulation.py` and `run_scripts_simulation_joint_train.py` | `reproduce_simulation_joint_freeze.py` | `figure_outputs/ds02_joint_SPCA_NN.png`, `figure_outputs/ds12_joint_SPCA_NN.png` |
| Figure 13 | `run_scripts_simulation_more_nonlinear.py` | `reproduce_simulation_more_nonlinear.py` | `figure_outputs/ds12_table_bf2.png` |
| Figure 14 | `run_scripts_simulation_more_nonlinear.py` | `reproduce_simulation_more_nonlinear.py` | `figure_outputs/ds12_table_bf3.png` |

The scripts also save long-format aggregated data and seed-count tables in `table_outputs/`.
