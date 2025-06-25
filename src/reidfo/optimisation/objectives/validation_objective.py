from typing import Dict, Any
from abc import abstractmethod
import datetime as dt
from hyperopt import STATUS_OK, STATUS_FAIL

from .base_objective import BaseObjective
from src.reidfo.logging import setup_logger

logger = setup_logger(__name__)


class ValidationObjective(BaseObjective):
    def __init__(self,
                 series_name: str,
                 start_date: dt.date,
                 val_start_date: dt.date,
                 test_start_date: dt.date,
                 end_date: dt.date,
                 n_regimes: int = 2,
                 seed: int = 42,):
        BaseObjective.__init__(self, series_name, start_date, end_date, seed)
        self.val_start_date = val_start_date
        self.test_start_date = test_start_date
        self.n_regimes = n_regimes
        self.rm_collector = None
        self.loss = None

    def _return_data(self, hyperparams: Dict[str, Any], loss_cls):
        try:
            self.loss = loss_cls(hyperparams, self.val_start_date, self.test_start_date)
            return {
                'loss': self.loss,
                'status': STATUS_OK,
                'hyperparams': hyperparams,
                'series': self.series_name
            }
        except Exception as e:
            logger.error(f"Error during optimisation: {type(e)}: {e}")
            return {
                'loss': 1e10,
                'status': STATUS_FAIL,
                'hyperparams': hyperparams,
                'series': self.series_name,
                'exception': str(e)
            }

    @abstractmethod
    def __call__(self, hyperparams: Dict[str, Any]) -> Dict:
        pass

    @abstractmethod
    def _validate_keys(self, hyperparams: Dict[str, Any]):
        """
        Ensure the necessary keys are present in the hyperparams

        :param hyperparams: Hyperparams selected by hyperopt
        :return:
        """

    @abstractmethod
    def clone(self, series_name: str) -> "BaseObjective":
        """
        Creates clone of the same class but with a different series
        :param series_name: Series name
        :return: BaseObjective
        """