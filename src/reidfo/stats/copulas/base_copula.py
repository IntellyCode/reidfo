import os
from typing import Optional, Union, List, Tuple, Dict
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.distributions.copula.api import ArchimedeanCopula
from src.reidfo.core.validation_utils import (
    check_axis_is_date,
    check_axis_is_string,
    check_df_for_nans,
)
from .abstract import AbstractCopula


class BaseCopula(AbstractCopula):
    def __init__(self, data: pd.DataFrame, seed: int = 42) -> None:
        """
        Base implementation of AbstractCopula that handles input validation
        and a common scatter‐plot routine. Concrete subclasses must implement `fit()`.

        :param data: DataFrame; each row represents a time series (index labels).
        :raises TypeError: if data is not a DataFrame.
        :raises ValueError: if fewer than two series (rows) are provided.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if data.shape[0] < 2:
            raise ValueError("DataFrame must have at least two rows to model a copula.")

        check_axis_is_date(data, axis=1)
        check_axis_is_string(data, axis=0)
        check_df_for_nans(data)

        self.data: pd.DataFrame = data.T
        self.fitted = False
        self._copulas = {}
        self.seed = seed

        self._uniforms: Dict[str, pd.Series] = {}
        self._simulations: Dict[Tuple[str, str], np.ndarray] = {}
        self._pairs = list(combinations(self.data.columns, 2))

    def fit(self):
        """
        Abstract: subclasses must
          1. compute & store self._uniforms[row_label],
          2. fit self.copula_ (a statsmodels Copula instance),
          3. set self.fitted = True.
        """
        raise NotImplementedError("Subclasses must implement `fit()`.")

    def plot(self,
             pairs: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None,
             n_samples: int = 10000,
             save_path: Optional[str] = None,
             show: bool = True
             ) -> Union[plt.Figure, List[plt.Figure]]:
        """
        Scatter plot of empirical or simulated copula uniforms.

        :param pairs: tuple or list thereof; None → all combinations.
        :param n_samples: when simulated=True, number of points to draw.
        :param save_path: filepath prefix to save figures.
        :param show: Whether to call plt.show().
        :returns: A Figure or list of Figures.
        """
        if not self.fitted:
            raise RuntimeError("Call `fit()` before plotting.")

        pair_list = self._normalize_pairs(pairs)

        figs = []
        for pair in pair_list:
            copula: ArchimedeanCopula = self._copulas[tuple(pair)]
            fig, _ = copula.plot_scatter(nobs=n_samples, random_state=self.seed)
            ax = fig.axes[0]
            ax.set_xlabel(pair[0])
            ax.set_ylabel(pair[1])
            if save_path:
                filename = f"{self.__class__.__name__}_{pair[0]}_vs_{pair[1]}.png"
                fig.savefig(os.path.join(save_path, filename))
            figs.append(fig)

        if show:
            for f in figs:
                f.show()

        return figs if len(figs) > 1 else figs[0]

    def _normalize_pairs(self, pairs: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]]) -> List[Tuple]:
        """Turn input into a list of 2‐tuples; generate all combos if None/empty."""
        if isinstance(pairs, tuple | list) and len(pairs) == 2:
            self._validate_pairs([pairs])
            return [tuple(pairs)]
        if isinstance(pairs, list) and pairs:
            self._validate_pairs(pairs)
            return [tuple(pair) for pair in pairs]
        if pairs is None or (isinstance(pairs, list) and not pairs):
            return self._pairs

        raise ValueError("`pairs` must be a tuple, list of tuples, or None.")

    def _validate_pairs(self, pair_list: List[Tuple]) -> None:
        """Ensure each pair is valid."""
        for r1, r2 in pair_list:
            if not (isinstance(r1, str) and isinstance(r2, str)):
                raise ValueError(f"Invalid pair elements: {r1!r}, {r2!r}")
            if r1 not in self.data.columns or r2 not in self.data.columns:
                raise ValueError(f"No empirical data for series {r1!r} or {r2!r}.")
            if (r1, r2) not in self._copulas.keys():
                raise ValueError(f"No copula for series {r1!r} and {r2!r}.")

