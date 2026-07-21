# Generalized Factor Neural Network Replication Package

This directory contains the code, data snapshots, and experiment entry points
used to replicate the empirical and simulation results for
**Generalized Factor Neural Network Model for High-dimensional Regression**.

Run the commands below from the repository root. The repository root is the
directory that contains `config.py`, `data/`, and `experiment/`.

When publishing this package on GitHub, upload the contents of this directory
as the repository root. Do not add another enclosing `replication/` directory;
all commands below are intentionally relative to `config.py`.

## Structure

```text
.
├── config.py
├── environment_versions.txt
├── requirements.txt
├── logs.py
├── data/
│   ├── ETF_price/
│   │   ├── README.md
│   │   └── SP Global/
│   │       ├── AllETFSimpleRet 10yrs.csv
│   │       ├── build_spglobal_cs_dataset.py
│   │       ├── cs_SimpleRet1-5-10-20-126_yLags1-20_SP500 as Y.csv.gz
│   │       └── notes.txt
│   ├── FRED-MD/
│   ├── covariate_standardized.py
│   ├── fast_data_standardized.py
│   ├── fredmd_data.py
│   └── univariate_funcs.py
├── experiment/
│   ├── ETF/
│   ├── FRED/
│   └── simulation/
├── methods/
├── models/
├── utils/
└── logs/
```

Experiment outputs are written under `logs/`.

## Environment

The experiments were run with Python 3.10 in a Linux environment with NVIDIA
GPUs. A fresh virtual environment or conda environment can be created from the
dependency list below.

### Computing Environment

The project was developed and run on Linux x86_64 systems with NVIDIA GPUs. A
principal local development environment had the following configuration:

- OS: Linux `6.8.0-90-generic` on `x86_64`
- Distribution family: Ubuntu GNU/Linux
- GPU: `NVIDIA GeForce RTX 2080 Ti`, `NVIDIA GeForce RTX 3090`
- CUDA version reported by `nvidia-smi`: `13.1`

The standardized 10-seed simulation reruns supplied as intermediary results
were run on RTX 3090 devices under Ubuntu 22.04. Their exact reference package
versions and GPU stack are recorded separately below.

Python dependencies are listed in:

```text
requirements.txt
```

An exact, anonymized snapshot of the principal software and GPU versions used
for the supplied RTX 3090 simulation results is provided in:

```text
environment_versions.txt
```

`requirements.txt` remains the portable installation specification;
`environment_versions.txt` documents the reference environment used for the
saved simulation outputs.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Create the log directory if needed:

```bash
mkdir -p logs
```

## Data

### Simulation Data

The simulation setup largely follows the original `FAST-NN` repository at
[wmyw96/FAST-NN](https://github.com/wmyw96/FAST-NN).

### FRED Data

The folder `data/FRED-MD/` contains the FRED-MD related files used
by the replication code. The underlying data source is
[FRED](https://fred.stlouisfed.org/), and the processing details follow the
original `FAST-NN` repository at
[wmyw96/FAST-NN](https://github.com/wmyw96/FAST-NN).

### ETF Data From S&P Global

The ETF/S&P 500 empirical section uses S&P Global sector and S&P 500
price-return index series. The archived local snapshot is:

```text
data/ETF_price/SP Global/AllETFSimpleRet 10yrs.csv
```

The processed dataset used by the experiment scripts is:

```text
data/ETF_price/SP Global/cs_SimpleRet1-5-10-20-126_yLags1-20_SP500 as Y.csv.gz
```

Additional source and processing notes are in:

```text
data/ETF_price/README.md
data/ETF_price/SP Global/notes.txt
```

## ETF / S&P 500 Sector-Index Experiment

The reported ETF application forecasts next-day S&P 500 price-index returns
using S&P Global sector price-return index predictors. The main input file is:

```text
data/ETF_price/SP Global/cs_SimpleRet1-5-10-20-126_yLags1-20_SP500 as Y.csv.gz
```

This file is generated from the archived S&P Global snapshot:

```text
data/ETF_price/SP Global/AllETFSimpleRet 10yrs.csv
```

The snapshot was downloaded from S&P Global on 2024-08-10. Since the S&P Global
website provides a rolling recent-ten-year download, this local snapshot is
included for exact replication, subject to the relevant data-license terms.

To rebuild the ETF cross-sectional dataset:

```bash
python "data/ETF_price/SP Global/build_spglobal_cs_dataset.py"
```

To run the ETF experiment entry point:

```bash
python experiment/ETF/run_scripts_ETF.py
```

The script is configured for the S&P Global sector-index dataset with 504
training days, 60 validation days, 252 test days, and test start date
2017-08-01.

The package does **not** require the legacy `SPDR constituents from CRSP/`
datasets for the reported ETF results. Those files are intentionally excluded
because they are not used by the current experiment and may be subject to
WRDS/CRSP redistribution restrictions.

## FRED-MD Experiment

The FRED-MD files are stored in:

```text
data/FRED-MD/
```

To run the FRED experiment entry point:

```bash
python experiment/FRED/run_scripts_FRED.py
```

The FRED scripts may be computationally intensive because they iterate over
many target series and model classes.

## Simulation Experiments

The main simulation entry point is:

```bash
python experiment/simulation/run_scripts_simulation.py
```

Additional simulation variants:

```bash
python experiment/simulation/run_scripts_simulation_ablation.py
python experiment/simulation/run_scripts_simulation_joint_train.py
python experiment/simulation/run_scripts_simulation_more_nonlinear.py
```

Dedicated scripts for Figures 2--4 and Figure 9 are also included. Their exact
paper settings, output files, and execution order are documented in:

```text
experiment/simulation/README.md
```

## Saved Intermediary Outputs

Clean saved-output archives may be supplied separately from the GitHub code so
that every paper table and figure can be audited without repeating expensive
neural-network training. Extract them into the following locations:

```text
logs/Simulation/reviewer_clean_results_package/
logs/FRED/reviewer_clean_results_package/
logs/ETF/reviewer_clean_results_package/
```

Each archive includes a `README_clean_run.txt` with exact commands. The
experiment-specific READMEs map those outputs to the numbered paper items. The
simulation archive includes saved inputs for Figures 2--4 and the 30 Optuna
studies used to regenerate Figure 9.

## Notes For Git Packaging

Python cache files such as `__pycache__/` and `*.pyc` should not be committed.
They are ignored by the repository-level `.gitignore`.

For the ETF replication package, the notebooks previously used during data
exploration are not required; the reproducible data-processing path is the
standalone script `build_spglobal_cs_dataset.py`.
