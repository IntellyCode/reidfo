from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd

from reidfo.core.validation_utils import check_index_is_datetime


class BaseDistance(ABC):
    def __init__(self, regime_labels: Dict[str, pd.Series]):
        """
        Abstract base for pairwise distances between regime label series.

        :param regime_labels: Dict of regime label Series keyed by time series name, each with a datetime index.
        """
        self.regime_labels = regime_labels
        self.distance_matrix = None
        for series in self.regime_labels.values():
            check_index_is_datetime(series)

    @abstractmethod
    def get_distance_matrix(self, window: int) -> pd.DataFrame:
        """
        Compute a pairwise distance matrix between the regime label series.

        :param window: Size of the window (in index steps) used when comparing series.
        :return: DataFrame of pairwise distances, indexed and columned by series name.
        """
