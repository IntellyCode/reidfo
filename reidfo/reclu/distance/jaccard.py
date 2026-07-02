from typing import Dict

import pandas as pd

from .base import BaseDistance


class JaccardDistance(BaseDistance):
    def __init__(self, regime_labels: Dict[str, pd.Series]):
        """
        :param regime_labels: Dict of regime label Series keyed by time series name, each with a datetime index.
        """
        super().__init__(regime_labels)
        self.regime_changes = {
            key: series.shift(-1).sub(series).iloc[:-1]
            for key, series in self.regime_labels.items()
        }

    def get_distance_matrix(self, window: int) -> pd.DataFrame:
        """
        Compute the Jaccard distance matrix between regime label series.

        For each pair of series, regime-change points in one series are matched against
        regime-change points of matching sign in the other series within `window` steps.

        :param window: Size of the window (in index steps) to search for matching shift signs.
        :return: DataFrame of pairwise Jaccard distances, indexed and columned by series name.
        """
        keys = list(self.regime_changes.keys())
        distance_matrix = pd.DataFrame(index=keys, columns=keys, dtype=float)
        for i, key_i in enumerate(keys):
            series_i = self.regime_changes[key_i]
            for j, key_j in enumerate(keys):
                if j > i:
                    continue
                if j == i:
                    distance_matrix.loc[key_i, key_j] = 0
                    continue
                series_j = self.regime_changes[key_j]
                jaccard_distance = self._pairwise_distance(series_i, series_j, window)
                distance_matrix.loc[key_i, key_j] = jaccard_distance
                distance_matrix.loc[key_j, key_i] = jaccard_distance
        self.distance_matrix = distance_matrix
        return self.distance_matrix

    @staticmethod
    def _pairwise_distance(series_i: pd.Series, series_j: pd.Series, window: int) -> float:
        series_i_index = series_i[series_i != 0].index
        series_j_index = series_j[series_j != 0].index
        positions_i = series_i.index.get_indexer(series_i_index)
        unique_length = len(series_i_index.union(series_j_index))

        overlap = 0
        for position in positions_i:
            change = series_i.iloc[position]
            window_start = max(position - window, 0)
            j_values = series_j.iloc[window_start:position + window + 1]
            overlap += (j_values.values == change).sum()

        jaccard = overlap / unique_length
        return 1 - jaccard
