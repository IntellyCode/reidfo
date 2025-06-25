import datetime as dt
from typing import Dict, Any

from .regime_loss import RegimeLoss
from src.reidfo.reid import StatisticalJumpModel
from src.reidfo.investing import Investor, LongShortStrategy


class SjmLambdaLoss(RegimeLoss):
    def __call__(self, hyperparams: Dict[str, Any], val_start: dt.date, test_start: dt.date) -> float:
        model = StatisticalJumpModel(self.data.get_series(),
                                     self.data.get_feature_matrix(),
                                     cont=False,
                                     prob=False,
                                     jump_penalty=hyperparams["regime_model"]['lam'],
                                     n_regimes=self.n_regimes,
                                     seed=self.seed)

        model.fit()
        labels = model.predict(self.validation_series_data.get_feature_matrix())
        if model.get_training_labels().nunique() <= 1:
            return 1e10
        investor = Investor(self.data.get_series(), model.get_training_labels(), self.validation_series_data.get_series())
        sharpe_dict = investor.invest(LongShortStrategy, labels)
        return -sharpe_dict["sharpe"]



