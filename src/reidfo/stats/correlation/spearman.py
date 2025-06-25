import pandas as pd
from scipy.stats import spearmanr

from .base import BaseCorrelation


class SpearmanCorrelation(BaseCorrelation):
    def __init__(self, df: pd.DataFrame):
        """
        Abstract base for pairwise correlations.
        Expects input DataFrame with rows as time series (index labels) and columns as time stamps.
        """
        super().__init__(df)

    def _compute_pair(self, x: pd.Series, y: pd.Series) -> float:
        return spearmanr(x, y).statistic
