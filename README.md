# reidfo

`reidfo` is a small Python package for regime-switching and forecasting workflows. The current codebase provides time-series data preparation, feature engineering, statistical diagnostics, copula fitting, plotting, and train/validation/test splitting utilities around that workflow.

The current package is source-installable from this repository. It is not documented here as a PyPI package, production-ready system, stable public API, benchmarked implementation, or finance-specific correctness layer.

## Install

Use a Python version compatible with the package metadata and install from the repository root:

```bash
git clone <repo-url>
cd reidfo
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Runtime dependencies are declared in `pyproject.toml`: `jumpmodels`, `loguru`, `matplotlib`, `scikit-learn`, `pandas`, `statsmodels`, and `nolds`.

## Data Contracts

Most APIs operate on pandas objects:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "A": [100.0, 101.0, 103.0, 102.0, 104.0],
        "B": [50.0, 49.0, 51.0, 52.0, 53.0],
    },
    index=pd.date_range("2024-01-01", periods=5, freq="D"),
)
```

Core expectations:

- Statistical diagnostics and copulas expect a `pd.DataFrame` with a datetime-like index and string column names.
- Copula classes require at least two columns and reject NaNs.
- Correlation, normality, stationarity, and general-statistics helpers drop NaNs per series where implemented.
- `DataSplitting` accepts a `pd.Series` or `pd.DataFrame`, rejects NaNs, and returns a dictionary mapping each column to `[train, val, test]` series.
- `FeatureEngineer` expects a time-indexed DataFrame with one named series per column. By default it computes percentage changes before collecting features unless `original=True` is passed.

See [Data Contracts](docs/data-contracts.md) for the longer version.

## Quickstart

```python
import pandas as pd

from reidfo.core.data_splitting import DataSplitting
from reidfo.feature_engineering.feature_engineer import FeatureEngineer
from reidfo.feature_engineering.collector.half_life_collector import HalfLifeCollector
from reidfo.stats.general_statistics import GeneralStatistics
from reidfo.stats.correlation.pearson import PearsonCorrelation
from reidfo.stats.copulas import ClaytonCopula

df = pd.DataFrame(
    {
        "A": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
        "B": [50.0, 49.0, 51.0, 52.0, 53.0, 54.0],
    },
    index=pd.date_range("2024-01-01", periods=6, freq="D"),
)

splits = DataSplitting(df).split(0.5, 0.25)

stats = GeneralStatistics(df).compute()
pearson = PearsonCorrelation(df).compute_matrix()

engineer = FeatureEngineer(df, clipper=None, scaler=None)
features = engineer.get_data(
    "A",
    HalfLifeCollector({"halflives": [2]}),
)

copula = ClaytonCopula(df)
copula.fit()
fig = copula.plot(pairs=("A", "B"), show=False)
```

`FeatureEngineer` raises when the collected feature matrix contains NaNs. Rolling-window collectors often create initial NaNs, so trim the date range or provide collectors/parameters that produce a complete matrix for the requested interval.

## API Overview

- `reidfo.core.preprocessing`: date filtering, standard-deviation clipping, and pandas-preserving standard scaling.
- `reidfo.core.dataframe_conversions`: index conversion to pandas datetime values.
- `reidfo.core.validation_utils`: validators for datetime indexes, string columns, NaNs, and regime counts.
- `reidfo.core.data_splitting.DataSplitting`: proportion-based and date-label-based train/validation/test splitting.
- `reidfo.core.plot`: time-series, cumulative-return, and regime-shading plots.
- `reidfo.core.validation_utils.validate_minimum_regimes`: validation for hard regime-label counts.
- `reidfo.feature_engineering`: `FeatureEngineer`, `TimeSeriesData`, and collector classes for half-life, rolling-window, and custom feature functions.
- `reidfo.stats`: descriptive statistics, Pearson/Spearman/Kendall correlation matrices, normality tests, stationarity diagnostics, and copula estimators.

Detailed pages are available under [docs](docs/).

## Development

Install the package in editable mode, then run the test suite from the repository root:

```bash
python -m pip install -e .
python -m pytest
```

Plot tests use a non-interactive Matplotlib backend in the test files. Some plotting helpers write PDF or PNG files to a supplied output directory.

## Status and Limitations

The package version is currently `0.0.0`. Public documentation is intentionally conservative because the repository does not yet define a formal compatibility policy, license text, benchmark suite, or domain-validation report.

Known limitations from the current code:

- Package-level `reidfo.__init__` does not re-export the main classes; import from submodules.
- The documented source tree does not expose a top-level forecasting estimator or regime-switching model class; it exposes supporting utilities and regime plotting/validation helpers.
- Several classes cache computed results on the instance.
- `FeatureEngineer` applies clipping before scaling when both are enabled.
- Copula plotting requires `fit()` first and saves PNG files when `save_path` is provided.
- `KPSS.plot()` is a no-op.
