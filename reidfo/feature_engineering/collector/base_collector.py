from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd


# reviewed
class BaseCollector(ABC):
    def __init__(self, params: Dict[str, Any]):
        """
        :param params: Parameters for the feature engineering function.
        """
        self.feat_params = params

    @abstractmethod
    def collect(self, time_series: pd.Series) -> pd.DataFrame:
        """
        Return a DataFrame where:
        - rows = time (aligned with input series)
        - columns = engineered feature names
        :param time_series: input time series
        """
        pass
