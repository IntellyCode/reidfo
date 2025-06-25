import pandas as pd
from scipy.stats import kendalltau

from .base import BaseCorrelation


class KendallTauCorrelation(BaseCorrelation):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def _compute_pair(self, x: pd.Series, y: pd.Series) -> float:
        return kendalltau(x, y).statistic
