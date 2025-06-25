from abc import abstractmethod
from typing import Dict
import pandas as pd


class BaseDistance:
    def __init__(self, regime_labels: Dict[str, pd.Series]):
        """
        A BaseDistance class.

        :param regime_labels: a dict of regime labels keyed by time series
        """
        self.regime_labels = regime_labels
        self.distance_matrix = None

    @abstractmethod
    def get_distance_matrix(self, window: int):
        """
        Creates a distance matrix
        """