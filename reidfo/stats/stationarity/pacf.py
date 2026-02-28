import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import _plot_corr

from .base import BaseStationarityTest


class PACF(BaseStationarityTest):
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame indexed by lag with one column per input time series.
            Uses cached results if already computed.
        """
        if self.scores is not None and getattr(self, "_pacf_values", None) is not None:
            return self.scores

        pacf_values = {}
        pacf_confint = {}
        max_len = 1

        for col, ts in self._iter_clean_series():
            if len(ts) < 2:
                pacf_values[col] = None
                pacf_confint[col] = None
                continue
            nlags = int(len(ts) * 0.4)
            pacf_vals, confint = pacf(ts, nlags=nlags, method='ywm', alpha=0.05)[:2]
            pacf_values[col] = pacf_vals
            pacf_confint[col] = confint
            max_len = max(max_len, len(pacf_vals))

        scores = pd.DataFrame(index=range(max_len), columns=self.df.columns, dtype=float)
        for col, values in pacf_values.items():
            if values is None:
                continue
            scores.loc[scores.index[:len(values)], col] = values
        scores.index.name = "lag"

        self.scores = scores
        self._pacf_values = pacf_values
        self._pacf_confint = pacf_confint
        return self.scores

    def plot(self, path: str, show: bool = False) -> None:
        """
        :param path: Directory where plots will be saved.
        :param show: If True, display plots interactively.
        """
        os.makedirs(path, exist_ok=True)

        if self.scores is None or getattr(self, "_pacf_values", None) is None:
            self.compute()
        scores = self.scores
        for col in scores.columns:
            values = self._pacf_values.get(col)
            if values is None or len(values) < 2:
                continue
            confint = self._pacf_confint.get(col)
            lags = np.arange(len(values))
            fig, ax = plt.subplots()
            _plot_corr(
                ax,
                "Partial Autocorrelation",
                values,
                confint,
                lags,
                irregular=False,
                use_vlines=True,
                vlines_kwargs={},
                auto_ylims=False,
            )
            plt.savefig(os.path.join(path, f"PACF_{col}.pdf"))
            if show:
                plt.show()
            plt.close()
