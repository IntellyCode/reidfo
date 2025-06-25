import datetime as dt
from typing import Dict, Any, Optional, Literal

from sklearn.metrics import accuracy_score

from .regime_loss import RegimeLoss
from src.reidfo.reid import StatisticalJumpModel
from src.reidfo.feature_engineering import TimeSeriesData
from src.reidfo.logging import setup_logger
from src.reidfo.refo import XGBoostModel

logger = setup_logger(__name__)

class HybridLoss(RegimeLoss):
    def __init__(self,
                 time_series_data: TimeSeriesData,
                 validation_series_data: TimeSeriesData,
                 n_regimes: int = 2,
                 sort_by: Optional[Literal["cumret", "vol", "freq", "ret", "mean"]] = 'cumret',
                 seed: int = 42):
        super().__init__(time_series_data, validation_series_data, n_regimes, seed)
        self.sort_by = sort_by
        logger.debug(f"Hybrid loss initialised:\n"
                     f"n_regimes={n_regimes}\n"
                     f"sort_by={self.sort_by}")

    def __call__(self, hyperparams: Dict[str, Any], val_start: dt.date, test_start: dt.date) -> float:
        logger.debug(f"Calling Hybrid Loss:\n"
                     f"hyperparams={hyperparams}\n"
                     f"val_start={val_start}\n"
                     f"test_start={test_start}")
        rm_data = self.data + self.validation_series_data.trim(1, -1)
        logger.debug(f"Merged Time Series Data")
        jm = StatisticalJumpModel(rm_data.get_series(),
                                  rm_data.get_feature_matrix(),
                                  n_regimes=self.n_regimes,
                                  seed=self.seed,
                                  jump_penalty=hyperparams["regime_model"]["lam"],
                                  sort_by=self.sort_by)
        logger.debug(f"Initialized Jump Model:\n"
                     f"n_regimes={self.n_regimes}\n"
                     f"seed={self.seed}\n"
                     f"lam={hyperparams['regime_model']['lam']}\n"
                     f"sort_by={self.sort_by}")
        jm.fit()
        logger.debug(f"Jump Model fitted")
        forecasting_model = XGBoostModel(self.data.get_feature_matrix(),
                                         jm.get_training_labels().loc[self.data.get_feature_matrix().index],
                                         hyperparams=None,
                                         smoothing_halflife=hyperparams["forecasting_model"]["smoothing_halflife"],
                                         seed=self.seed)
        logger.debug(f"Initialized XGBoost Model")
        forecasting_model.fit()
        logger.debug(f"XGBoost Model fitted")
        xgboost_regime_labels = forecasting_model.predict(self.validation_series_data.get_feature_matrix())
        logger.debug(f"XGBoost Regime Labels")

        pred_labels = xgboost_regime_labels
        test_labels = jm.get_training_labels().loc[pred_labels.index]
        accuracy = accuracy_score(test_labels, pred_labels)
        return -accuracy
