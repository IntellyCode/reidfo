import pandas as pd
from typing import Union

from src.reidfo.core.validation_utils import check_axis_is_date, check_axis_is_string, check_df_for_nans
from .base_strategy import BaseStrategy

from src.reidfo.logging import setup_logger
logger = setup_logger(__name__)


# reviewed
class Investor:
    def __init__(self,
                 train_returns: Union[pd.Series, pd.DataFrame],
                 train_labels: Union[pd.Series, pd.DataFrame],
                 returns: Union[pd.Series, pd.DataFrame],
                 ):
        """
        Stores return data and training labels for later strategy execution.

        :param train_returns: Series or DataFrame of training returns.
        :param train_labels: Regime labels aligned with training returns.
        :param returns: Return series or DataFrame to apply the strategy on.
        """
        self.train_returns = self._to_dataframe(train_returns)
        self.train_labels = self._to_dataframe(train_labels)
        self.returns = self._to_dataframe(returns)
        self._validate_dimensions(self.train_returns, self.train_labels)
        logger.debug(f"Initialised Investor Class")

    def invest(self,
               strategy: type[BaseStrategy],
               labels: Union[pd.Series, pd.DataFrame],
               ) -> dict:
        """
        Applies the strategy to stored returns using the given labels.

        :param labels: Regime labels aligned with the stored return series.
        :param strategy: A strategy class (not an instance), subclass of BaseStrategy.
        :return: Dictionary with:
                 - 'investments': Series or DataFrame of adjusted returns
                 - 'sharpe': Sharpe ratio of the investment performance
        """
        labels_df = self._to_dataframe(labels)
        self._validate_dimensions(self.returns, labels_df)

        result_df = self.returns.apply(
            lambda row: strategy(
                train_returns=self.train_returns.loc[row.name],
                train_labels=self.train_labels.loc[row.name]
            )(row, labels_df.loc[row.name]),
            axis=1
        )

        return {
            "investments": result_df,
            "sharpe": self._calculate_sharpe(result_df),
        }

    @staticmethod
    def _calculate_sharpe(df: pd.DataFrame) -> pd.DataFrame:
        mean_return = df.mean(axis=1)
        std_return = df.std(axis=1)
        sharpe_ratio = mean_return / std_return
        sharpe_ratio = sharpe_ratio.replace([float("inf"), -float("inf")], -1e10).fillna(-1e6)
        if len(sharpe_ratio.index) == 1:
            return sharpe_ratio.iloc[0]
        return sharpe_ratio

    @staticmethod
    def _to_dataframe(obj: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(obj, pd.Series):
            return obj.to_frame(name="series").T
        return obj

    @staticmethod
    def _validate_dimensions(returns: pd.DataFrame, labels: pd.DataFrame) -> None:
        check_df_for_nans(returns)
        check_df_for_nans(labels)

        check_axis_is_string(returns, axis=0)
        check_axis_is_string(labels, axis=0)

        check_axis_is_date(returns, axis=1)
        check_axis_is_date(labels, axis=1)

        if not returns.columns.equals(labels.columns):
            raise ValueError("Returns and labels must have the same column order\n"
                             f"Returns: {returns.columns}\n"
                             f"Labels: {labels.columns}")

        if set(returns.index) != set(labels.index):
            raise ValueError("Returns and labels must have the same row labels\n"
                             "Returns: {returns.index}\n"
                             "Labels: {labels.index}")
