import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from reidfo.core.validation_utils import check_index_is_datetime, check_columns_are_strings


class BaseCorrelation(ABC):
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Abstract base for pairwise correlations.
        Expects input DataFrame with rows as time entries (datetime index) and columns as series names (strings).

        :param df: DataFrame with datetime index and string column names.
        """
        self.df = df
        self.scores = None
        check_index_is_datetime(self.df)
        check_columns_are_strings(self.df)

    @staticmethod
    def _align_series(s1: pd.Series, s2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Restrict both series to their maximum common set of timestamps.

        :param s1: First series (already NaN-dropped, indexed by timestamps).
        :param s2: Second series (already NaN-dropped, indexed by timestamps).
        :return: Tuple of (s1, s2) restricted to the intersection of their timestamp indices.
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
        Build a correlation matrix for the specified column pairs in `df`.

        :param labels1: Column labels for the matrix rows. Defaults to all columns.
        :param labels2: Column labels for the matrix columns. Defaults to all columns.
        :return: DataFrame of shape (len(labels1), len(labels2)) with pairwise correlation scores.
        """
        full_matrix = labels1 is None and labels2 is None
        if full_matrix and self.scores is not None:
            return self.scores
        if labels1 is None:
            labels1 = list(self.df.columns)
        if labels2 is None:
            labels2 = list(self.df.columns)
        matrix = pd.DataFrame(index=labels1, columns=labels2, dtype=float)
        if full_matrix:
            cleaned = {col: self.df[col].dropna() for col in labels1}
            for i, c1 in enumerate(labels1):
                for j, c2 in enumerate(labels2):
                    if j < i:
                        matrix.at[c1, c2] = matrix.at[c2, c1]
                    else:
                        a1, a2 = self._align_series(cleaned[c1], cleaned[c2])
                        matrix.at[c1, c2] = self._compute_pair(a1, a2)
            self.scores = matrix
        else:
            for c1 in labels1:
                for c2 in labels2:
                    s1, s2 = self.df[c1].dropna(), self.df[c2].dropna()
                    a1, a2 = self._align_series(s1, s2)
                    matrix.at[c1, c2] = self._compute_pair(a1, a2)
        return matrix
