# API Reference

This page is a compact map of documented modules and public classes/functions visible in the current source tree.

## Core

- `reidfo.core.dataframe_conversions.convert_index_to_datetime(df)`
- `reidfo.core.data_splitting.DataSplitting(data).split(train, val=None)`
- `reidfo.core.preprocessing.filter_date_range(obj, start_date=None, end_date=None)`
- `reidfo.core.preprocessing.clip_by_std(df, mul=3.0)`
- `reidfo.core.preprocessing.standard_scale(df)`
- `reidfo.core.validation_utils.check_index_is_datetime(obj)`
- `reidfo.core.validation_utils.check_columns_are_strings(obj)`
- `reidfo.core.validation_utils.check_df_for_nans(obj)`
- `reidfo.core.validation_utils.validate_minimum_regimes(labels, required=2)`

## Plotting

- `reidfo.core.plot.plot_time_series(data, start_date=None, end_date=None, ax=None, ylabel_data="Variable")`
- `reidfo.core.plot.plot_prodret(ret_series, start_date=None, end_date=None, ax=None, ylabel="Cumulative Returns")`
- `reidfo.core.plot.plot_regimes(regimes, start_date=None, end_date=None, ax=None, regime_colors=("g", "r"), regime_labels=("Bull", "Bear"))`

## Feature Engineering

- `reidfo.feature_engineering.feature_engineer.FeatureEngineer`
- `reidfo.feature_engineering.time_series_data.TimeSeriesData`
- `reidfo.feature_engineering.collector.base_collector.BaseCollector`
- `reidfo.feature_engineering.collector.half_life_collector.HalfLifeCollector`
- `reidfo.feature_engineering.collector.windowed_collector.WindowedCollector`
- `reidfo.feature_engineering.collector.custom_collector.CustomCollector`

Feature functions are listed in `reidfo.feature_engineering.functional_dictionary.functional_dictionary`.

## Statistics

- `reidfo.stats.general_statistics.GeneralStatistics`
- `reidfo.stats.correlation.pearson.PearsonCorrelation`
- `reidfo.stats.correlation.spearman.SpearmanCorrelation`
- `reidfo.stats.correlation.kendall.KendallTauCorrelation`
- `reidfo.stats.normality.jarque_bera.JarqueBeraTest`
- `reidfo.stats.normality.shapiro.ShapiroWilkTest`
- `reidfo.stats.stationarity.acf.ACF`
- `reidfo.stats.stationarity.pacf.PACF`
- `reidfo.stats.stationarity.kpss.KPSS`
- `reidfo.stats.stationarity.hurst.Hurst`

## Copulas

- `reidfo.stats.copulas.ClaytonCopula`
- `reidfo.stats.copulas.GumbelCopula`
- `reidfo.stats.copulas.StudentTCopula`
- `reidfo.stats.copulas.BaseCopula`
