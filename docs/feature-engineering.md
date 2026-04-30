# Feature Engineering

Feature engineering is built around `FeatureEngineer`, collector classes, and `TimeSeriesData`.

## FeatureEngineer

```python
from reidfo.feature_engineering.feature_engineer import FeatureEngineer
from reidfo.feature_engineering.collector.half_life_collector import HalfLifeCollector

engineer = FeatureEngineer(df, clipper=None, scaler=None)
data = engineer.get_data("A", HalfLifeCollector({"halflives": [5, 20]}))
```

By default, `FeatureEngineer` computes percentage changes with `df.pct_change(fill_method=None).iloc[1:]` and passes the selected return series to the collector. Pass `original=True` to collect features from the original column instead.

The collected feature matrix is filtered by `start_date` and `end_date`, then passed through the configured clipper and scaler. Defaults are `clip_by_std` followed by `standard_scale`. Set either argument to `None` to disable it.

## Collectors

Available collectors:

- `HalfLifeCollector`: exponentially weighted mean, log downside deviation, and exponentially weighted Sortino ratio for each configured half-life.
- `WindowedCollector`: observation, absolute change, previous absolute change, and rolling window features such as mean, standard deviation, left/right half statistics, and related windowed features.
- `CustomCollector`: builds features from names in `functional_dictionary`.

`CustomCollector` expects parameters shaped like:

```python
from reidfo.feature_engineering.collector.custom_collector import CustomCollector

collector = CustomCollector({
    "feat_params": {
        "function_list": ["obs", "ewm_me", "cen_std"],
        "halflives": [5],
        "windows": [6],
    }
})
```

Rolling and lagged functions can produce NaNs at the start of a series. `FeatureEngineer.get_data()` raises if NaNs remain after filtering, clipping, and scaling, so choose a later `start_date` or use parameters that produce complete features for the selected interval.
