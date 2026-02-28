import pandas as pd
from scipy.stats import spearmanr

from .base import BaseCorrelation


class SpearmanCorrelation(BaseCorrelation):
    def __init__(self, df: pd.DataFrame):
        """
        :param df: DataFrame with datetime index and string column names.
        """
        super().__init__(df)

    def _compute_pair(self, x: pd.Series, y: pd.Series) -> float:
        """
        :param x: First aligned series.
        :param y: Second aligned series.
        :return: Spearman rank correlation coefficient.
        """
        return spearmanr(x, y).statistic
