import datetime as dt
from typing import Dict, Any
from functools import partial

from src.reidfo.hybrid_model import HybridModel
from src.reidfo.refo import XGBoostModel
from src.reidfo.reid import StatisticalJumpModel
from .base_loss import BaseLoss
from ...feature_engineering import TimeSeriesData


class SjmXGBoostLoss(BaseLoss):
    def __init__(self,
                 time_series_data: TimeSeriesData,
                 forecasting_time_series_data: TimeSeriesData,
                 n_regimes: int = 2,
                 seed: int = 42):
        super().__init__(time_series_data, seed)
        self.forecasting_data = forecasting_time_series_data
        if not self.forecasting_data.get_series().index.equals(self.data.get_series().index):
            raise ValueError("Index must be the same for forecasting data and data \n"
                             f"Forecasting index: {self.forecasting_data.get_series().index} \n"
                             f"Data index: {self.data.get_series().index}")
        self.n_regimes = n_regimes

    def __call__(self,
                 hyperparams: Dict[str, Any],
                 val_start: dt.date,
                 test_start: dt.date) -> float:
        model = HybridModel(self.data,
                            partial(StatisticalJumpModel,
                                    n_regimes=self.n_regimes,
                                    cont=False,
                                    prob=False,
                                    jump_penalty=hyperparams["regime_model"].get("lam", 0),
                                    seed=self.seed),
                            partial(XGBoostModel,
                                    seed=self.seed,
                                    smoothing_halflife=hyperparams["forecasting_model"].get("smoothing_halflife", None)),
                            self.forecasting_data)
        model.split(train=val_start, val=test_start)
        model.fit(val=False)
        labels = model.predict(val=True)
        sharpe = model.invest(labels)
        return -sharpe


