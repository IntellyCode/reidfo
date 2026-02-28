from abc import ABC, abstractmethod
import pandas as pd

from reidfo.core.validation_utils import check_columns_are_strings, check_index_is_datetime


class BaseNormalityTest(ABC):
    def __init__(self, df: pd.DataFrame):
        """
        :param df: DataFrame with index as timestamps and columns as time series names.
            Index must be datetime-like and column labels must be strings.
        """
        self.df = df
        self.scores = None
        check_index_is_datetime(self.df)
        check_columns_are_strings(self.df)

    def _iter_clean_series(self):
        """
        :returns: Iterator of (column_name, numpy_array) with NaNs dropped.
        """
        for col in self.df.columns:
            ts = self.df[col].dropna().values
            yield col, ts

    @abstractmethod
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame of test results per column (time series)
        """
        pass
