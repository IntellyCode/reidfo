import os

import nolds
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .base import BaseStationarityTest


class Hurst(BaseStationarityTest):
    def __init__(self, df: pd.DataFrame):
        """
        :param df: DataFrame with index as time and columns as time series names.
            Index must be datetime-like and column labels must be strings.
        """
        super().__init__(df)
        self._plot_data = {}

    def compute(self) -> pd.DataFrame:
        """
        :returns: DataFrame with one column "Hurst_exponent" for each column (time series).
            Uses cached results if already computed.
        """
        if self.scores is not None:
            return self.scores

        hurst_dict = {}
        plot_data = {}
        for col, ts in self._iter_clean_series():
            hurst_value, debug_data = self._compute_hurst_and_plot_data(ts)
            hurst_dict[col] = hurst_value
            plot_data[col] = debug_data
        self.scores = pd.DataFrame.from_dict(
            hurst_dict,
            orient="index",
            columns=["Hurst_exponent"],
        )
        self._plot_data = plot_data
        return self.scores

    @staticmethod
    def _compute_hurst_and_plot_data(
        ts: np.ndarray,
    ) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
        """
        :param ts: 1D array of time series values with NaNs removed.
        :returns: Tuple of (Hurst exponent, plot data). Plot data contains
            (nvals, rsvals, poly) for the log-log fit, or None when there are
            fewer than 2 values.
        """
        if len(ts) < 2:
            return np.nan, None
        hurst, debug_data = nolds.hurst_rs(ts, debug_data=True)
        return float(hurst), debug_data

    def plot(self, path: str, show: bool = False) -> None:
        """
        :param path: Directory where plots will be saved.
        :param show: If True, display plots interactively in addition to saving.
            Uses cached debug data from compute().
        """
        os.makedirs(path, exist_ok=True)

        if self.scores is None:
            self.compute()

        for col, value in self.scores["Hurst_exponent"].items():
            if np.isnan(value):
                continue
            debug_data = self._plot_data.get(col)
            if debug_data is None:
                continue
            nvals, rsvals, poly = debug_data
            fig, ax = plt.subplots()
            ax.plot(nvals, rsvals, "o", label="log(R/S)")
            ax.plot(nvals, poly[0] * nvals + poly[1], "-", label="fit")
            ax.set_xlabel("log(n)")
            ax.set_ylabel("log((R/S)_n)")
            plot_path = os.path.join(path, f"Hurst_{col}.pdf")
            fig.savefig(plot_path)
            if show:
                plt.show()
            plt.close(fig)
