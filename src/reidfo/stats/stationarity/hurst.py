import os
import numpy as np
import pandas as pd
import nolds

from .base import BaseStationarityTest


class Hurst(BaseStationarityTest):
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame with one column 'H' (Hurst exponent) for each time series
        """
        hurst_dict = {}

        for idx in self.df.index:
            ts = self.df.loc[idx].dropna().values
            if len(ts) < 2:
                hurst_dict[idx] = np.nan
                continue

            H = nolds.hurst_rs(ts)
            hurst_dict[idx] = H

        self.scores = pd.DataFrame.from_dict(hurst_dict, orient='index', columns=['H'])
        return self.scores

    def plot(self, path: str, show: bool = False) -> None:
        """
        :param path: Ignored for Hurst (no plots generated)
        :param show: Ignored for Hurst (no plots generated)
        """
        pass
