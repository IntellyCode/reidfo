import numpy as np
import pandas as pd
from scipy.stats import shapiro

from .base import BaseNormalityTest


class ShapiroWilkTest(BaseNormalityTest):
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame with Shapiro-Wilk test statistic and p-value per time series
        """
        results = {}

        for idx in self.df.index:
            ts = self.df.loc[idx].dropna().values
            if len(ts) < 3:
                results[idx] = [np.nan, np.nan]
                continue

            res = shapiro(ts)
            results[idx] = [res.statistic, res.pvalue]

        self.scores = pd.DataFrame.from_dict(results, orient='index', columns=['Shapiro-Wilk', 'p-value'])
        return self.scores
