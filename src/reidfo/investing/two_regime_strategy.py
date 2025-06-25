from abc import ABC
import pandas as pd

from src.reidfo.core.validation_utils import validate_minimum_regimes
from .base_strategy import BaseStrategy


# reviewed
class TwoRegimeStrategy(BaseStrategy, ABC):
    def __init__(self, train_returns: pd.Series, train_labels: pd.Series):
        """
        Initializes the green regime based on the highest mean return in training data.

        :param train_returns: Training return series.
        :param train_labels: Training regime labels aligned with the returns.
        """
        validate_minimum_regimes(train_labels, required=2)
        self.green_regime = self._infer_green_regime(train_returns, train_labels)

    def _infer_green_regime(self, returns: pd.Series, labels: pd.Series) -> int:
        """
        Validates that there are exactly 2 regimes and selects the bullish one.

        :return: Label of the regime with the highest mean return.
        """
        df = self._prepare(returns, labels)
        return df.groupby('regime')['return'].mean().idxmax()
