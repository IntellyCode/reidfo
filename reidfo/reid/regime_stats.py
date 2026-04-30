from typing import Union

import pandas as pd

from reidfo.core.validation_utils import check_index_is_datetime


class RegimeStats:
    def __init__(self,
                 time_series: Union[pd.Series, pd.DataFrame],
                 labels: Union[pd.Series, pd.DataFrame],
                 returns: bool = True):
        """
        Aggregate values per regime.

        Two input layouts are accepted:

        - ``Series`` × ``Series``: index is datetime; aggregation is over a single series.
        - ``DataFrame`` × ``DataFrame``: rows are entities, columns are datetime stamps;
          aggregation runs row-by-row and results are concatenated with a MultiIndex.

        :param time_series: Series or DataFrame of values.
        :param labels: Regime labels, matching the shape and index of ``time_series``.
        :param returns: If True, also report cumulative and per-period scaled returns per regime.
        :raises ValueError: If shape, index, or column alignment fails.
        :raises TypeError: If ``time_series`` and ``labels`` are not both Series or both DataFrames.
        """
        self.time_series = time_series
        self.labels = labels
        self.returns = returns

        if isinstance(self.time_series, pd.Series) and isinstance(self.labels, pd.Series):
            check_index_is_datetime(self.time_series)
            check_index_is_datetime(self.labels)
            if not self.time_series.index.equals(self.labels.index):
                raise ValueError("Index mismatch between time series and labels.")
        elif isinstance(self.time_series, pd.DataFrame) and isinstance(self.labels, pd.DataFrame):
            # Columns are the datetime axis in this layout; rows are entity identifiers.
            if not pd.api.types.is_datetime64_any_dtype(self.time_series.columns) \
                    or not pd.api.types.is_datetime64_any_dtype(self.labels.columns):
                raise ValueError("DataFrame columns must be a datetime index.")
            if set(self.time_series.index) != set(self.labels.index):
                raise ValueError("Index mismatch between time series and labels.")
            if not self.time_series.columns.equals(self.labels.columns):
                raise ValueError("Column mismatch between time series and label DataFrame.")
        else:
            raise TypeError("time_series and labels must both be Series or both be DataFrames.")

    def get_regime_stats(self) -> pd.DataFrame:
        """
        Compute per-regime ``mean``, ``std``, ``count``, and (when ``returns=True``)
        ``cumret`` and ``scaled_cumret``.

        :return: Flat DataFrame for Series input; MultiIndex-column DataFrame for DataFrame input.
        """
        if isinstance(self.time_series, pd.Series):
            return self._aggregate_series(self.time_series, self.labels)

        result = {
            str(row): self._aggregate_series(self.time_series.loc[row, :], self.labels.loc[row, :])
            for row in self.time_series.index
        }
        return pd.concat(result.values(), axis=1, keys=result.keys())

    def _aggregate_series(self, ts: pd.Series, lb: pd.Series) -> pd.DataFrame:
        aligned = pd.DataFrame({"value": ts.to_numpy(), "regime": lb.to_numpy()})
        grouped = aligned.groupby("regime")["value"]
        stats = grouped.agg(["mean", "std", "count"])

        if self.returns:
            stats["cumret"] = grouped.apply(lambda x: (1 + x).prod() - 1)
            stats["scaled_cumret"] = (1 + stats["cumret"]) ** (1 / stats["count"]) - 1

        return stats
