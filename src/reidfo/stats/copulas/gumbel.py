import pandas as pd
from scipy.stats import kendalltau
from statsmodels.distributions.copula.api import GumbelCopula as SMGumbel

from .base_copula import BaseCopula


class GumbelCopula(BaseCopula):
    def __init__(self, data: pd.DataFrame, seed: int = 42):
        super().__init__(data, seed)

    def fit(self):
        self.fitted = True
        for pair in self._pairs:
            tau, _ = kendalltau(self.data[pair[0]], self.data[pair[1]])
            theta = 1 / (1 - tau)
            if theta < 1:
                print(f"For {pair} theta  < 1 hence GumbelCopula is not appropriate. Skipping {pair}!")
                continue
            cp = SMGumbel(theta=theta)
            self._copulas[pair] = cp


