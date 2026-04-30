# Statistics

The statistics modules operate column-wise on pandas DataFrames with datetime-like indexes and string column names unless noted otherwise.

## Descriptive Statistics

```python
from reidfo.stats.general_statistics import GeneralStatistics

stats = GeneralStatistics(df).compute()
```

`GeneralStatistics.compute()` returns one row per input column with `Mean`, `Median`, `StdDev`, `Min`, `Max`, `Skew`, and `Kurt`. NaNs are dropped per column.

## Correlation

```python
from reidfo.stats.correlation.pearson import PearsonCorrelation
from reidfo.stats.correlation.spearman import SpearmanCorrelation
from reidfo.stats.correlation.kendall import KendallTauCorrelation

pearson = PearsonCorrelation(df).compute_matrix()
spearman = SpearmanCorrelation(df).compute_matrix()
kendall = KendallTauCorrelation(df).compute_matrix()
```

`compute_matrix()` returns a DataFrame indexed and columned by series names. Optional `labels1` and `labels2` arguments compute a rectangular subset. Full matrices are cached on the instance.

## Normality

```python
from reidfo.stats.normality.jarque_bera import JarqueBeraTest
from reidfo.stats.normality.shapiro import ShapiroWilkTest

jb = JarqueBeraTest(df).compute()
sw = ShapiroWilkTest(df).compute()
```

Both classes return one row per input column with a test statistic, p-value, and boolean `normal` field using `pvalue > 0.05`. Columns with fewer than three non-NaN values return NaNs.

## Stationarity and Dependence Diagnostics

```python
from reidfo.stats.stationarity.acf import ACF
from reidfo.stats.stationarity.pacf import PACF
from reidfo.stats.stationarity.kpss import KPSS
from reidfo.stats.stationarity.hurst import Hurst

acf_scores = ACF(df).compute()
pacf_scores = PACF(df).compute()
kpss_scores = KPSS(df).compute()
hurst_scores = Hurst(df).compute()
```

`ACF` and `PACF` return lag-indexed DataFrames and can write PDF plots. `KPSS` returns `KPSS_stat`, `p_value`, and `stationary`; its `plot()` method does not generate output. `Hurst` returns `Hurst_exponent` and can write log-log fit plots for computable series.
