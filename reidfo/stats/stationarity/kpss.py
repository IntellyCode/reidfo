import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import kpss

from .base import BaseStationarityTest


class KPSS(BaseStationarityTest):
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame with columns [KPSS_stat, p_value, stationary] for each column (time series).
            Uses cached results if already computed.
        """
        if self.scores is not None:
            return self.scores

        kpss_dict = {}
        for col, ts in self._iter_clean_series():
            if len(ts) < 5:
                kpss_dict[col] = [np.nan, np.nan, np.nan]
                continue
            try:
                stat, pval, _, _ = kpss(ts, regression='c', nlags='auto')
                kpss_dict[col] = [stat, pval, pval > 0.05]
            except Exception:
                kpss_dict[col] = [np.nan, np.nan, np.nan]

        self.scores = pd.DataFrame.from_dict(
            kpss_dict,
            orient='index',
            columns=['KPSS_stat', 'p_value', 'stationary'],
        )
        return self.scores

    def plot(self, path: str, show: bool = False) -> None:
        """
        :param path: Ignored for KPSS (no plots generated).
        :param show: Ignored for KPSS (no plots generated).
        """
        pass
