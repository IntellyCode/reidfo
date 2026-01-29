import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import _plot_corr

from .base import BaseStationarityTest


class ACF(BaseStationarityTest):
    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame indexed by lag with one column per input time series.
        """
        if self.scores is not None and getattr(self, "_acf_values", None) is not None:
            return self.scores

        acf_values = {}
        acf_confint = {}
        max_len = 1

        for col, ts in self._iter_clean_series():
            if len(ts) < 2:
                acf_values[col] = None
                acf_confint[col] = None
                continue

            nlags = self._nlags(ts)
            acf_vals, confint = acf(ts, nlags=nlags, fft=True, alpha=0.05)[:2]
            acf_values[col] = acf_vals
            acf_confint[col] = confint
            max_len = max(max_len, len(acf_vals))

        scores = pd.DataFrame(index=range(max_len), columns=self.df.columns, dtype=float)
        for col, values in acf_values.items():
            if values is None:
                continue
            scores.loc[scores.index[:len(values)], col] = values
        scores.index.name = "lag"
        self.scores = scores
        self._acf_values = acf_values
        self._acf_confint = acf_confint
        return self.scores

    def _nlags(self, ts: np.ndarray) -> int:
        """
        :param ts: Time series values.
        :returns: Number of lags to use for ACF calculations.
        """
        return len(ts) - 1

    def plot(self, path: str, show: bool = False) -> None:
        """
        :param path: Directory where plots will be saved.
        :param show: If True, display plots interactively.
        """
        os.makedirs(path, exist_ok=True)

        scores = self.compute()
        for col in scores.columns:
            values = self._acf_values.get(col)
            if values is None or len(values) < 2:
                continue
            confint = self._acf_confint.get(col)
            lags = np.arange(len(values))
            fig, ax = plt.subplots()
            _plot_corr(
                ax,
                "Autocorrelation",
                values,
                confint,
                lags,
                irregular=False,
                use_vlines=True,
                vlines_kwargs={},
                auto_ylims=False,
            )
            plt.savefig(os.path.join(path, f"ACF_{col}.pdf"))
            if show:
                plt.show()
            plt.close()
