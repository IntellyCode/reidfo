import pandas as pd
from statsmodels.distributions.copula.api import ClaytonCopula as SMClayton

from .base_copula import BaseCopula


class ClaytonCopula(BaseCopula):
    def __init__(self, data: pd.DataFrame, seed: int = 42):
        super().__init__(data, seed)

    def fit(self):
        self.fitted = True
        for pair in self._pairs:
            cp = SMClayton()
            theta = cp.fit_corr_param(self.data[list(pair)])
            cp = SMClayton(theta=theta)
            self._copulas[pair] = cp


