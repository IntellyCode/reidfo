# Plotting

Plotting helpers live under `reidfo.core.plot` and return Matplotlib axes or figures.

## Time Series

```python
from reidfo.core.plot.time_series import plot_time_series

ax = plot_time_series(series, ylabel_data="Value")
```

`plot_time_series` plots a pandas Series, optionally filtered by `start_date` and `end_date`. Pass an existing Matplotlib `Axes` with `ax=` to draw into it.

## Cumulative Returns

```python
from reidfo.core.plot.prodret import plot_prodret

ax = plot_prodret(return_series)
```

`plot_prodret` plots cumulative returns computed as `(1 + returns).cumprod() - 1` and formats the y-axis as percentages. If the first value is not zero, the helper prepends a zero return one inferred frequency step before the first index.

## Regime Shading

```python
from reidfo.core.plot.regimes import plot_regimes

ax = plot_regimes(regime_series)
```

`plot_regimes` shades contiguous blocks of regime labels across the y-axis. The default colors and labels assume two regimes, but both can be overridden.

## Matplotlib Settings

`reidfo.core.plot.util.matplotlib_setting()` changes global Matplotlib rcParams, including LaTeX text rendering. Use it only when those global settings are appropriate for the current process.
