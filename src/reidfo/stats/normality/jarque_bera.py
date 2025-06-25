import numpy as np
import pandas as pd
from scipy.stats import jarque_bera

from .base import BaseNormalityTest


class JarqueBeraTest(BaseNormalityTest):
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame with Jarque-Bera test statistic and p-value per time series
        """
        results = {}

        for idx in self.df.index:
            ts = self.df.loc[idx].dropna().values
            if len(ts) < 3:
                results[idx] = [np.nan, np.nan]
                continue

            res = jarque_bera(ts)
            results[idx] = [res.statistic, res.pvalue]

        self.scores = pd.DataFrame.from_dict(results, orient='index',columns=['Jarque-Bera', 'p-value'])
        return self.scores
