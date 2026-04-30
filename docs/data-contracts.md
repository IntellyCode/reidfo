# Data Contracts

`reidfo` uses pandas `Series` and `DataFrame` objects as its main input and output format for regime-switching and forecasting support code. The package generally validates at class construction or helper entry points, then assumes the data shape remains valid.

## Common Time-Series Frame

Statistical diagnostics and copula estimators expect a DataFrame shaped like this:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "A": [1.0, 2.0, 3.0],
        "B": [1.5, 2.5, 3.5],
    },
    index=pd.date_range("2024-01-01", periods=3, freq="D"),
)
```

The index must be datetime-like and all column labels must be strings for correlation, normality, stationarity, and copula classes. Use `reidfo.core.dataframe_conversions.convert_index_to_datetime` when an index contains date strings that pandas can parse.

## NaNs

NaN handling differs by area:

- `DataSplitting` rejects NaNs during initialization.
- Copula classes reject NaNs during initialization.
- General statistics, correlations, normality tests, and stationarity diagnostics drop NaNs per column before computing where implemented.
- `FeatureEngineer.get_data()` rejects any NaNs remaining in the collected feature matrix after date filtering, clipping, and scaling.

## Series and Feature Alignment

`TimeSeriesData` stores a `series` and `feature_matrix`. Their indexes must match exactly. The `+` and `+=` operations append compatible, non-overlapping `TimeSeriesData` blocks only when index types, feature columns, feature dtypes, and series dtype match.

## Splitting

`DataSplitting(data).split(train, val)` accepts either proportions or date labels:

- Float arguments split by integer positions using `int(n * proportion)`.
- Date arguments must be present in the index.
- `train` and `val` must use the same type when both are supplied.
- If `val` is omitted for a float split, the validation proportion defaults to the training proportion.
- If `val` is omitted for a date split, validation is empty and test starts at the train date.

## Regime Labels

`validate_minimum_regimes(labels, required=2)` checks that each supplied label series contains exactly the requested number of distinct regimes. `plot_regimes` expects a one-dimensional regime-label Series and shades contiguous regime blocks.
