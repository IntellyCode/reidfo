import pandas as pd
from scipy.stats import pearsonr

from .base import BaseCorrelation


class PearsonCorrelation(BaseCorrelation):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def _compute_pair(self, x: pd.Series, y: pd.Series) -> float:
        return pearsonr(x, y).statistic
