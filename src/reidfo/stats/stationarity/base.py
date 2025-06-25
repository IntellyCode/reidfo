import pandas as pd
from abc import ABC, abstractmethod

from src.reidfo.core.validation_utils import check_axis_is_date, check_axis_is_string


class BaseStationarityTest(ABC):
    def __init__(self, df: pd.DataFrame):
        """
        :param df: DataFrame with rows as time series and columns as timestamps
        """
        self.df = df
        self.scores = None
        check_axis_is_date(self.df, axis=1)
        check_axis_is_string(self.df, axis=0)

    @abstractmethod
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame containing test statistics for each row (time series)
        """
        pass

    @abstractmethod
    def plot(self, path: str, show: bool = False) -> None:
        """
        :param path: Path to folder where plots will be saved
        :param show: Whether to display plots interactively
        """
        pass
