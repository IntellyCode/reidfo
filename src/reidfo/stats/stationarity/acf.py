import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

from .base import BaseStationarityTest


class ACF(BaseStationarityTest):
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame where each row contains the full ACF vector of the corresponding time series
        """
        acf_dict = {}

        for idx in self.df.index:
            ts = self.df.loc[idx].dropna().values
            if len(ts) < 2:
                acf_dict[idx] = np.nan
                continue

            nlags = len(ts) - 1
            acf_vals = acf(ts, nlags=nlags, fft=True)
            acf_dict[idx] = acf_vals

        self.scores = pd.DataFrame.from_dict(acf_dict).T
        return self.scores

    def plot(self, path: str, show: bool = False) -> None:
        """
        :param path: Directory where plots will be saved
        :param show: If True, display plots interactively
        """
        os.makedirs(path, exist_ok=True)

        for idx in self.df.index:
            ts = self.df.loc[idx].dropna().values
            if len(ts) < 2:
                continue
            nlags = len(ts) - 1
            plot_acf(ts, lags=nlags)
            plt.xticks(np.arange(0, len(ts) + 1, step=20))
            plt.savefig(f"{path}/ACF_{idx}.png")
            if show:
                plt.show()
            plt.close()
