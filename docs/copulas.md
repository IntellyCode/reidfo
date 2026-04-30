# Copulas

Copula estimators live under `reidfo.stats.copulas` and wrap statsmodels copula implementations for pairwise fitting.

```python
from reidfo.stats.copulas import ClaytonCopula, GumbelCopula, StudentTCopula

copula = ClaytonCopula(df, seed=42)
copula.fit()
fig = copula.plot(pairs=("A", "B"), show=False)
```

## Input Requirements

Copula input must be a pandas DataFrame with:

- datetime-like index
- string column names
- at least two columns
- no NaNs

Every pair of columns is considered during initialization.

## Estimators

- `ClaytonCopula` estimates a parameter per pair with statsmodels' `fit_corr_param`.
- `GumbelCopula` derives theta from Kendall tau and skips pairs where theta is below the supported lower bound.
- `StudentTCopula` estimates degrees of freedom by optimizing a pseudo-likelihood objective, then derives the correlation matrix from transformed pseudo-observations.

## Plotting

Call `fit()` before `plot()`. The `pairs` argument can be `None`, a single pair like `("A", "B")`, or a list of pairs. When `save_path` is provided, PNG files are written with names such as `ClaytonCopula_A_vs_B.png`.

The plot method returns one Matplotlib figure for a single pair, a list of figures for multiple pairs, or an empty list when there are no fitted pairs to plot.
