import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from src.reidfo.core.validation_utils import (
    check_axis_is_date,
    check_axis_is_string,
)


class BaseCorrelation(ABC):
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Abstract base for pairwise correlations.
        Expects input DataFrame with rows as time series (index labels) and columns as time stamps.
        """
        self.df = df
        self._matrix = pd.DataFrame(index=df.index, columns=df.index, dtype=float)
        check_axis_is_date(self.df, axis=1)
        check_axis_is_string(self.df, axis=0)

    @staticmethod
    def _align_series(s1: pd.Series, s2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Restrict both series to their maximum common index (timestamps).
        """
        common_idx = s1.index.intersection(s2.index)
        return s1.loc[common_idx], s2.loc[common_idx]

    @abstractmethod
    def _compute_pair(self, x: pd.Series, y: pd.Series) -> float:
        """
        Compute the correlation coefficient between two aligned series.
        """

    def compute_matrix(self, labels1: Optional[list] = None, labels2: Optional[list] = None) -> pd.DataFrame:
        """
        Build an (n × n) correlation matrix for all row‐pairs in `df`.

        :param labels1: The labels of the rows in the matrix.
        :param labels2: The labels of the columns in the matrix.
        """
        if labels1 is None:
            labels1 = self.df.index
        if labels2 is None:
            labels2 = self.df.index
        for r1 in labels1:
            for r2 in labels2:
                s1 = self.df.loc[r1].dropna()
                s2 = self.df.loc[r2].dropna()
                a1, a2 = self._align_series(s1, s2)
                self._matrix.at[r1, r2] = self._compute_pair(a1, a2)
        return self._matrix

