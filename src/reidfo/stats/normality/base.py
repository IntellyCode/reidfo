import pandas as pd
from abc import ABC, abstractmethod


class BaseNormalityTest(ABC):
    def __init__(self, df: pd.DataFrame):
        """
        :param df: DataFrame with rows as time series and columns as timestamps
        """
        self.df = df
        self.scores = None

    @abstractmethod
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame of test results per row
        """
        pass
