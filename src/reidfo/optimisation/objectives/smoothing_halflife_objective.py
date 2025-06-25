from typing import Dict, Any

import datetime as dt

from .base_objective import BaseObjective
from .sjm_xgboost_objective import SjmXGBoostObjective


class SmoothingHalflifeObjective(SjmXGBoostObjective):
    def __init__(self,
                 series_name: str,
                 start_date: dt.date,
                 val_start_date: dt.date,
                 test_start_date: dt.date,
                 end_date: dt.date,
                 dfdl_params: dict,
                 n_regimes: int = 2,
                 seed: int = 42):
        super().__init__(series_name, start_date, val_start_date, test_start_date, end_date, n_regimes, seed)
        self.sjm_collector = None
        self.xgb_collector = None
        self.loss = None
        self.dfdl_params = dfdl_params

    def __call__(self, hyperparams: Dict[str, Any]) -> Dict:
        self.dfdl_params[self.series_name]["forecasting_model"].update(hyperparams["forecasting_model"])
        params = self.dfdl_params[self.series_name]
        return super().__call__(params)

    def clone(self, series_name: str) -> "BaseObjective":
        return SmoothingHalflifeObjective(
            series_name=series_name,
            start_date=self.start_date,
            val_start_date=self.val_start_date,
            test_start_date=self.test_start_date,
            end_date=self.end_date,
            n_regimes=self.n_regimes,
            dfdl_params=self.dfdl_params,
            seed=self.seed
        )
