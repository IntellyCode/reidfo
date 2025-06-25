from abc import ABC, abstractmethod
import pandas as pd


# reviewed
class BaseStrategy(ABC):
    @abstractmethod
    def __init__(self, train_returns: pd.Series, train_labels: pd.Series):
        """
        Optional initialization using training data. Subclasses may use or ignore it.

        :param train_returns: Series of returns from training data.
        :param train_labels: Series of regime labels aligned with training returns.
        """
        pass

    @abstractmethod
    def __call__(self, returns: pd.Series, labels: pd.Series) -> pd.Series:
        """
        Applies the strategy logic to a return series and label series.

        :param returns: Series of returns, indexed by time.
        :param labels: Corresponding regime labels, indexed identically to `returns`.
        :return: Series of strategy-adjusted returns.
        """
        pass

    @staticmethod
    def _prepare(returns: pd.Series, labels: pd.Series) -> pd.DataFrame:
        """
        Bundles return and label series for use in strategy application.

        :param returns: Return series to invest on.
        :param labels: Regime labels for the given return series.
        :return: DataFrame with ['return', 'regime'] columns.
        """
        return pd.DataFrame({'return': returns, 'regime': labels})