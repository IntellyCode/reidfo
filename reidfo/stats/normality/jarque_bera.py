import numpy as np
import pandas as pd
from scipy.stats import jarque_bera

from .base import BaseNormalityTest


class JarqueBeraTest(BaseNormalityTest):
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame with columns [Jarque-Bera, p-value, normal] for each column (time series).
            Uses cached results if already computed.
        """
        if self.scores is not None:
            return self.scores

        results = {}
        for col, ts in self._iter_clean_series():
            if len(ts) < 3:
                results[col] = [np.nan, np.nan, np.nan]
                continue
            res = jarque_bera(ts)
            results[col] = [res.statistic, res.pvalue, res.pvalue > 0.05]

        self.scores = pd.DataFrame.from_dict(
            results,
            orient='index',
            columns=['Jarque-Bera', 'p-value', 'normal'],
        )
        return self.scores
