from typing import Dict, Any, Optional, Literal
import datetime as dt

from .base_objective import BaseObjective
from .validation_objective import ValidationObjective
from src.reidfo.logging import setup_logger
from src.reidfo.feature_engineering.collector import CustomCollector
from src.reidfo.optimisation.loss import HybridLoss

logger = setup_logger(__name__)

class HybridObjective(ValidationObjective):
    def __init__(self,
                 series_name: str,
                 start_date: dt.date,
                 val_start_date: dt.date,
                 test_start_date: dt.date,
                 end_date: dt.date,
                 feat_params: Dict[str, Any],
                 sort_by: Optional[Literal["cumret", "vol", "freq", "ret", "mean"]] = 'cumret',
                 n_regimes: int = 2,
                 seed: int = 42, ):
        super().__init__(series_name, start_date, val_start_date, test_start_date, end_date, n_regimes, seed)
        self.feat_params = feat_params
        self.sort_by = sort_by
        logger.debug(f"HybridObjective Initialised with:\n"
                     f"feat_params={feat_params}\n"
                     f"n_regimes={n_regimes}\n"
                     f"series_name={series_name}\n"
                     f"sort_by={sort_by}")

    def __call__(self, hyperparams: Dict[str, Any]) -> Dict:
        rm, rf = self._validate_keys(hyperparams)
        self._validate_ready()
        self.collector = CustomCollector(self.feat_params)
        training_data = self.feature_engineer.get_data(self.series_name,
                                                       self.collector,
                                                       self.start_date,
                                                       self.val_start_date,
                                                       original=self.feat_params["original"])
        validation_data = self.feature_engineer.get_data(self.series_name,
                                                         self.collector,
                                                         self.val_start_date,
                                                         self.test_start_date,
                                                         original=self.feat_params["original"])
        logger.debug("Training data and validation data extracted:\n"
                     f"training_data_matrix={training_data.get_feature_matrix().shape}\n"
                     f"validation_data_matrix={validation_data.get_feature_matrix().shape}")
        loss_cls = HybridLoss(training_data, validation_data,self.n_regimes, self.sort_by, self.seed)
        return self._return_data(hyperparams, loss_cls)

    def _validate_keys(self, hyperparams: Dict[str, Any]):
        """
        Ensure the necessary keys are present in the hyperparams

        :param hyperparams: Hyperparams selected by hyperopt
        :return:
        """
        if "regime_model" not in hyperparams:
            raise KeyError("Missing 'regime_model' block in hyperparams.")

        if "forecasting_model" not in hyperparams:
            raise KeyError("Missing 'forecasting_model' block in hyperparams.")

        rm = hyperparams["regime_model"]
        if "lam" not in rm:
            raise KeyError("Missing 'lam' block in hyperparams regime model")

        rf = hyperparams["forecasting_model"]
        if "smoothing_halflife" not in rf:
            raise KeyError("Missing 'smoothing_halflife' block in hyperparams regime model")
        return rm, rf

    def clone(self, series_name: str) -> "BaseObjective":
        """
        Creates clone of the same class but with a different series
        :param series_name: Series name
        :return: BaseObjective
        """
        return HybridObjective(series_name,
                               start_date=self.start_date,
                               val_start_date=self.val_start_date,
                               test_start_date=self.test_start_date,
                               end_date=self.end_date,
                               feat_params=self.feat_params,
                               n_regimes=self.n_regimes,
                               seed=self.seed)