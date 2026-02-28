import pandas as pd
from loguru import logger
from statsmodels.distributions.copula.api import StudentTCopula as SMStudent
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from scipy.stats import t

from .base_copula import BaseCopula


class StudentTCopula(BaseCopula):
    def __init__(self, data: pd.DataFrame, seed: int = 42):
        super().__init__(data, seed)

    def fit(self):
        self.fitted = True
        for pair in self._pairs:
            df, corr = self._estimate_df(pair)
            cp = SMStudent(df=df, corr=corr)
            self._copulas[pair] = cp

    def _estimate_df(self, pair, init_df=5):
        """
        Estimates the degrees of freedom (df) for a 2D t-copula using PMLE.
        """
        data = self.data[list(pair)].to_numpy()
        data_clean = data[~np.isnan(data).any(axis=1)]
        u_data = np.array([
            rankdata(data_clean[:, i]) / (len(data_clean) + 1)
            for i in range(data_clean.shape[1])
        ]).T

        result = minimize(
            lambda df_array: self._df_objective(df_array, u_data),
            [init_df],
            bounds=[(2.01, 30)]
        )
        df_estimate = result.x[0]

        pseudo_obs_final = t.ppf(u_data, df=df_estimate)
        emp_corr_final = np.corrcoef(pseudo_obs_final.T)

        return df_estimate, emp_corr_final

    @staticmethod
    def _df_objective(df_array, u_data):
        """
        Objective function for optimizing degrees of freedom.
        """
        df_val = df_array[0]
        if df_val <= 2:
            return np.inf
        try:
            pseudo_obs = t.ppf(u_data, df=df_val)
            emp_corr = np.corrcoef(pseudo_obs.T)
            t_copula = SMStudent(corr=emp_corr, df=df_val)
            logpdf = np.log(t_copula.pdf(u_data))
            return -np.sum(logpdf)
        except Exception as e:
            logger.warning(f"Exception during optimisation of DF for StudentTCopula: {e}")
            return np.inf