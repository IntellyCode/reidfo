from typing import Dict, Any
import datetime as dt
from hyperopt import STATUS_OK, STATUS_FAIL

from src.reidfo.feature_engineering.collector import CustomCollector
from .validation_objective import ValidationObjective
from .base_objective import BaseObjective
from ..loss import SjmXGBoostLoss


class SjmXGBoostObjective(ValidationObjective):
    def __init__(self,
                 series_name: str,
                 start_date: dt.date,
                 val_start_date: dt.date,
                 test_start_date: dt.date,
                 end_date: dt.date,
                 n_regimes: int = 2,
                 seed: int = 42,):
        ValidationObjective.__init__(self, series_name, start_date, val_start_date, test_start_date, end_date, seed)
        self.n_regimes = n_regimes
        self.fm_collector = None
        self.loss = None

    def __call__(self, hyperparams: Dict[str, Any]) -> Dict:
        rm, rf = self._validate_keys(hyperparams)
        self._validate_ready()
        self.rm_collector = CustomCollector(rm)
        self.fm_collector = CustomCollector(rf)
        rm_data = self.feature_engineer.get_data(self.series_name, self.rm_collector, self.start_date, self.end_date)
        fm_data = self.feature_engineer.get_data(self.series_name, self.fm_collector, self.start_date, self.end_date)
        loss_cls = SjmXGBoostLoss(rm_data, fm_data, n_regimes=self.n_regimes, seed=self.seed)
        return self._return_data(hyperparams, loss_cls)

    def _validate_keys(self, hyperparams: Dict[str, Any]):
        """
        Ensures both model configurations exist and contain required fields.
        """
        if "regime_model" not in hyperparams:
            raise KeyError("Missing 'regime_model' block in hyperparams.")
        if "forecasting_model" not in hyperparams:
            raise KeyError("Missing 'forecasting_model' block in hyperparams.")

        rm = hyperparams["regime_model"]
        fm = hyperparams["forecasting_model"]

        if "feat_params" not in rm or "lam" not in rm:
            raise KeyError("Missing 'feat_params' or 'lam' in regime_model block.")

        if "feat_params" not in fm:
            raise KeyError("Missing 'feat_params' in forecasting_model block.")
        return rm, fm

    def clone(self, series_name: str) -> "BaseObjective":
        return SjmXGBoostObjective(
            series_name=series_name,
            start_date=self.start_date,
            val_start_date=self.val_start_date,
            test_start_date=self.test_start_date,
            end_date=self.end_date,
            n_regimes=self.n_regimes,
            seed=self.seed
        )
