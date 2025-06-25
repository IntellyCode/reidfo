from typing import Union
import pandas as pd

from src.reidfo.core.validation_utils import check_axis_is_date, check_axis_is_string


# reviewed
class RegimeStats:
    def __init__(self,
                 time_series: Union[pd.Series, pd.DataFrame],
                 labels: Union[pd.Series, pd.DataFrame],
                 returns: bool = True):
        """
        Initialize RegimeStats.

        :param time_series: A time series or dataframe to analyze. Index must be datetime.date or convertible.
        :param labels: A Series or DataFrame of regime labels with the same index (and columns if applicable).
        :param returns: If True, compute cumulative return per regime.
        :raises ValueError: If shape or index mismatch.
        """
        self.time_series = time_series
        self.labels = labels
        self.returns = returns

        if isinstance(self.time_series, pd.Series) and isinstance(self.labels, pd.Series):
            check_axis_is_date(self.time_series)
            check_axis_is_date(self.labels)
            if not self.time_series.index.equals(self.labels.index):
                raise ValueError("Index mismatch between time series and labels.")
        elif isinstance(self.time_series, pd.DataFrame) and isinstance(self.labels, pd.DataFrame):
            check_axis_is_date(self.time_series, axis=1)
            check_axis_is_date(self.labels, axis=1)
            if set(self.time_series.index) != set(self.labels.index):
                raise ValueError("Index mismatch between time_series and labels.")
            if not self.time_series.columns.equals(self.labels.columns):
                raise ValueError("Column mismatch between time series and label DataFrame.")
        else:
            raise TypeError("time_series and labels must both be Series or both be DataFrames.")

    def get_regime_stats(self) -> pd.DataFrame:
        """
        Computes per-regime summary statistics: mean, std, count, and optional cumulative return.

        :return: A DataFrame of regime stats. For Series input: flat. For DataFrame input: MultiIndex columns.
        """
        if isinstance(self.time_series, pd.Series):
            return self._aggregate_series(self.time_series, self.labels)

        result = {}
        for row in self.time_series.index:
            ts_row = self.time_series.loc[row, :]
            lb_row = self.labels.loc[row, :]
            result[str(row)] = self._aggregate_series(ts_row, lb_row)
        return pd.concat(result.values(), axis=1, keys=result.keys())

    def _aggregate_series(self, ts: pd.Series, lb: pd.Series) -> pd.DataFrame:
        """
        Aggregates a single series by regime.

        :param ts: Series of values (e.g. returns)
        :param lb: Series of regime labels
        :return: Aggregated stats for this single series
        """
        aligned = pd.DataFrame({'value': ts, 'regime': lb})
        grouped = aligned.groupby('regime')['value']
        stats = grouped.agg(['mean', 'std', 'count'])

        if self.returns:
            stats['cumret'] = grouped.apply(lambda x: (1 + x).prod() - 1)
            stats['scaled_cumret'] = (1 + stats['cumret']) ** (1 / stats['count']) - 1

        return stats
