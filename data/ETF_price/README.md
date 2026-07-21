# ETF price data notes

This folder contains the data inputs and processing scripts for the S&P 500
sector-index forecasting experiment in the replication package.

## Main dataset used in the paper

The reported ETF/S&P 500 empirical results use:

```text
SP Global/cs_SimpleRet1-5-10-20-126_yLags1-20_SP500 as Y.csv.gz
```

This file is generated from:

```text
SP Global/AllETFSimpleRet 10yrs.csv
```

The source series are S&P Global price-return index series, not total-return or
net-total-return series. The local notes record the S&P Global source page and
download date:

```text
SP Global/notes.txt
```

Because the S&P Global website provides a rolling recent-ten-year download, the
snapshot file `SP Global/AllETFSimpleRet 10yrs.csv` should be archived with the
replication package if redistribution is permitted by the data license. Without
this snapshot, a future re-download may not reproduce the same sample window.

To rebuild the main cross-sectional file from the S&P Global return snapshot:

```bash
python "data/ETF_price/SP Global/build_spglobal_cs_dataset.py"
```

This creates 55 sector-index return features based on 1, 5, 10, 20, and 126
trading-day rolling sums, appends `y_lag1` through `y_lag20`, and sets `y` to
the next-day S&P 500 price-index return.

## Excluded WRDS/CRSP files

Earlier internal checks used WRDS/CRSP ETF panels and SPDR constituent files.
Those files are not needed for the reported paper results and are intentionally
excluded from this replication package.

## Legacy CRSP constituent files

The folder:

```text
SPDR constituents from CRSP/
```

contains older CRSP constituent-based ETF datasets. The current ETF experiment
scripts do not read these files, and they are not required for the reported
S&P 500 sector-index results. They are large and may be subject to WRDS/CRSP
redistribution restrictions, so they should generally be excluded from a public
replication package unless there is a separate reason and permission to include
them.
