from typing import Dict
import pandas as pd

from .base import BaseDistance

class JaccardDistance(BaseDistance):
    def __init__(self, regime_labels: Dict[str, pd.Series]):
        super().__init__(regime_labels)
        self.regime_changes = {}
        for key, series in self.regime_labels.items():
            diff = series.shift(-1) - series
            self.regime_changes[key] = diff.iloc[:-1]


    def get_distance_matrix(self, window: int):
        """
        Compute Jaccard distance matrix between regime label series.

        :param window: int, size of the window (in index steps) to search for matching shift signs.
        :return: np.ndarray, 2D array with Jaccard distances between series.
        """
        distance_matrix = pd.DataFrame(columns=list(self.regime_changes.keys()), index=list(self.regime_changes.keys()))
        for i, (key_i, series_i) in enumerate(self.regime_changes.items()):
            for j, (key_j, series_j) in enumerate(self.regime_changes.items()):
                if j > i:
                    continue
                if j == i:
                    distance_matrix.loc[key_i, key_j] = 0
                    continue
                series_i_index = series_i[series_i != 0].index
                series_j_index = series_j[series_j != 0].index
                positions_i = series_i.index.get_indexer(series_i_index)
                unique_indices = series_i_index.union(series_j_index)
                unique_length = len(unique_indices)
                overlap = 0
                for index in positions_i:
                    i_index = series_i.iloc[index]
                    j_values = series_j.iloc[index-window:index+window+1]
                    matches = (j_values.values == i_index).sum()
                    overlap += matches

                jaccard = overlap / unique_length
                jaccard_distance = 1 - jaccard
                distance_matrix.loc[key_i, key_j] = jaccard_distance
                distance_matrix.loc[key_j, key_i] = jaccard_distance
        self.distance_matrix = distance_matrix
        return self.distance_matrix

