import datetime as dt
from abc import abstractmethod
from typing import Dict, Any

from .base_loss import BaseLoss
from src.reidfo.feature_engineering import TimeSeriesData


class RegimeLoss(BaseLoss):
    def __init__(self,
                 time_series_data: TimeSeriesData,
                 validation_series_data: TimeSeriesData,
                 n_regimes: int = 2,
                 seed: int = 42):
        super().__init__(time_series_data, seed)
        self.validation_series_data = validation_series_data
        self.n_regimes = n_regimes

    @abstractmethod
    def __call__(self, hyperparams: Dict[str, Any], val_start: dt.date, test_start: dt.date) -> float:
        """
        :param hyperparams: Hyperparams selected by hyperopt
        :param val_start: The date representing the start of the validation period
        :param test_start: The date representing the start of the test period
        :return: Value to be minimised
        """