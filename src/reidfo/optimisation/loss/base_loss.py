from abc import ABC, abstractmethod
from typing import Dict, Any
import datetime as dt

from src.reidfo.feature_engineering import TimeSeriesData


class BaseLoss(ABC):
    """
    Loss class to encapsulate the full model evaluation logic including feature loading, model fitting,
    regime prediction, and applying an investing strategy.
    """
    def __init__(self,
                 time_series_data: TimeSeriesData,
                 seed: int = 42):
        """
        :param time_series_data: An instance of a TimeSeriesData object
        :param seed: Random seed
        """
        self.data = time_series_data
        self.seed = seed

    @abstractmethod
    def __call__(self,
                 hyperparams: Dict[str, Any],
                 val_start: dt.date,
                 test_start: dt.date) -> float:
        """
        :param hyperparams: Hyperparams selected by hyperopt
        :param val_start: The date representing the start of the validation period
        :param test_start: The date representing the start of the test period
        :return: Value to be minimised
        """

