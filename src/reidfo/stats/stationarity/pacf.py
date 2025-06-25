import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf

from .base import BaseStationarityTest


class PACF(BaseStationarityTest):
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame where each row contains the PACF vector of the corresponding time series
        """
        pacf_dict = {}
        for idx in self.df.index:
            ts = self.df.loc[idx].dropna().values
            if len(ts) < 2:
                pacf_dict[idx] = np.nan
                continue

            nlags = int(len(ts) * 0.4)
            pacf_vals = pacf(ts, nlags=nlags)
            pacf_dict[idx] = pacf_vals

        self.scores = pd.DataFrame.from_dict(pacf_dict).T
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
            nlags = int(len(ts) * 0.4)
            plot_pacf(ts, lags=nlags, method='ywm')
            plt.xticks(np.arange(0, len(ts) + 1, step=20))
            plt.savefig(f"{path}/PACF_{idx}.png")
            if show:
                plt.show()
            plt.close()
