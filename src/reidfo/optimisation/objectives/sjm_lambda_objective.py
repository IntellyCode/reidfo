from typing import Dict, Any
import datetime as dt

from .sjm_xgboost_objective import SjmXGBoostObjective
from .base_objective import BaseObjective
from src.reidfo.feature_engineering.collector import CustomCollector
from ..loss.sjm_lambda_loss import SjmLambdaLoss


class SjmLambdaObjective(SjmXGBoostObjective):
    def __init__(self,
                 series_name: str,
                 start_date: dt.date,
                 val_start_date: dt.date,
                 test_start_date: dt.date,
                 end_date: dt.date,
                 feat_params: Dict[str, Any],
                 n_regimes: int = 2,
                 seed: int = 42):
        """
        Optimised lambda for a statistical jump model during out of sample investing

        :param series_name: Series name (ease of use)
        :param start_date: Start date
        :param val_start_date: Start of validation period
        :param test_start_date: Start of testing period
        :param end_date: Dnd date
        :param feat_params: Feature Parameters obtained by hyperopt optimisation of regime model features
        :param n_regimes: Number of regimes
        :param seed: Random seed
        """
        SjmXGBoostObjective.__init__(self, series_name, start_date, val_start_date, test_start_date, end_date, n_regimes, seed)
        self.collector = None
        self.feat_params = feat_params

    def clone(self, series_name: str) -> "BaseObjective":
        return SjmLambdaObjective(series_name,
                                  start_date=self.start_date,
                                  val_start_date=self.val_start_date,
                                  test_start_date=self.test_start_date,
                                  end_date=self.end_date,
                                  feat_params=self.feat_params,
                                  n_regimes=self.n_regimes,
                                  seed=self.seed)

    def _validate_keys(self, hyperparams: Dict[str, Any]):
        if "regime_model" not in hyperparams:
            raise KeyError("Missing 'regime_model' block in hyperparams.")

        rm = hyperparams["regime_model"]
        if "lam" not in rm:
            raise KeyError("Missing 'lam' block in hyperparams regime model")
        return rm

    def __call__(self, hyperparams: Dict[str, Any]) -> Dict:
        self._validate_keys(hyperparams)
        self._validate_ready()
        self.collector = CustomCollector(self.feat_params)
        training_data = self.feature_engineer.get_data(self.series_name, self.collector, self.start_date, self.val_start_date)
        validation_data = self.feature_engineer.get_data(self.series_name, self.collector, self.val_start_date, self.test_start_date)
        loss_cls = SjmLambdaLoss(training_data, validation_data, n_regimes=self.n_regimes, seed=self.seed)
        return self._return_data(hyperparams, loss_cls)

