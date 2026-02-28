import numpy as np
import pandas as pd
from scipy.stats import shapiro

from .base import BaseNormalityTest


class ShapiroWilkTest(BaseNormalityTest):
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame with columns [Shapiro-Wilk, p-value, normal] for each column (time series).
            Uses cached results if already computed.
        """
        if self.scores is not None:
            return self.scores

        results = {}
        for col, ts in self._iter_clean_series():
            if len(ts) < 3:
                results[col] = [np.nan, np.nan, np.nan]
                continue
            res = shapiro(ts)
            results[col] = [res.statistic, res.pvalue, res.pvalue > 0.05]

        self.scores = pd.DataFrame.from_dict(
            results,
            orient='index',
            columns=['Shapiro-Wilk', 'p-value', 'normal'],
        )
        return self.scores
