from typing import Dict, Any
from abc import ABC, abstractmethod
import datetime as dt


class AbstractObjective(ABC):
    def __init__(self,
                 series_name: str,
                 start_date: dt.date,
                 end_date: dt.date,
                 seed: int):
        self.series_name = series_name
        self.feature_engineer = None
        self.start_date = start_date
        self.end_date = end_date
        self.time_series = None
        self.seed = seed

    @abstractmethod
    def __call__(self, hyperparams: Dict[str, Any]) -> Dict:
        pass

    @abstractmethod
    def _validate_keys(self, hyperparams: Dict[str, Any]):
        """
        Ensure the necessary keys are present in the hyperparams

        :param hyperparams: Hyperparams selected by hyperopt
        :return:
        """

