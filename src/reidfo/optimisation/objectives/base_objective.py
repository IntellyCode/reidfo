from abc import abstractmethod
from typing import Dict, Any
import pandas as pd
import datetime as dt

from .abstract import AbstractObjective
from src.reidfo.feature_engineering import FeatureEngineer


class BaseObjective(AbstractObjective):
    def __init__(self,
                 series_name: str,
                 start_date: dt.date,
                 end_date: dt.date,
                 seed: int):
        AbstractObjective.__init__(self, series_name, start_date, end_date, seed)

    def set_series(self, time_series: pd.Series):
        self.feature_engineer = FeatureEngineer(time_series.to_frame(self.series_name).T)

    def _validate_ready(self):
        if self.feature_engineer is None:
            raise ValueError("Feature engineering must be set before using the objective.")

    @abstractmethod
    def __call__(self, hyperparams: Dict[str, Any]) -> Dict:
        """self._validate_ready()
        return self._validate_keys(hyperparams)"""

    @abstractmethod
    def _validate_keys(self, hyperparams: Dict[str, Any]):
        """
        Ensure the necessary keys are present in the hyperparams

        :param hyperparams: Hyperparams selected by hyperopt
        :return:
        """

    @abstractmethod
    def clone(self, series_name: str) -> "BaseObjective":
        """
        Creates clone of the same class but with a different series
        :param series_name: Series name
        :return: BaseObjective
        """
