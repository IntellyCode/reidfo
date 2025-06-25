import pandas as pd
from typing import Callable, Optional, Union, Type
import datetime as dt

from src.reidfo.reid.abstract import RegimeModel
from src.reidfo.refo.forecasting_model import ForecastingModel
from src.reidfo.core import DataSplitting
from src.reidfo.feature_engineering import TimeSeriesData
from src.reidfo.investing import Investor, LongShortStrategy, BaseStrategy
from ..logging import setup_logger
logger = setup_logger("reidfo.hybrid_model")


# reviewed
class HybridModel:
    def __init__(self,
                 data: TimeSeriesData,
                 regime_model_factory: Callable[..., RegimeModel],
                 forecasting_model_factory: Callable[..., ForecastingModel],
                 forecasting_data: Optional[TimeSeriesData] = None
                 ):
        """
        Initializes the hybrid model with data and model factories.

        :param data: A TimeSeriesData object containing the return series and feature matrix.
        :param regime_model_factory: Callable that returns a RegimeModel when passed training features.
        :param forecasting_model_factory: Callable that returns a ForecastingModel when passed training features and regime labels.
        :param forecasting_data: A TimeSeriesData object containing the return series and feature matrix for the forecasting
        """
        self.data = data
        self.forecasting_data = forecasting_data if forecasting_data is not None else self.data

        if not self.forecasting_data.get_series().index.equals(self.data.get_series().index):
            raise ValueError("Index must be the same for forecasting data and data \n"
                             f"Forecasting index: {self.forecasting_data.get_series().index} \n"
                             f"Data index: {self.data.get_series().index}")

        self.regime_model_factory = regime_model_factory
        self.forecasting_model_factory = forecasting_model_factory
        self._val_exists = False
        self._trained_with_val = False
        self._trained = False
        self._splits = {}

        self.regime_model = None
        self.forecasting_model = None
        self.investment_results = None
        logger.info("Initialised Hybrid Model")

    def split(self, train: Union[float, dt.date], val: Optional[Union[float, dt.date]] = None) -> None:
        """
        Splits the series and feature matrix into training, validation (optional), and test sets.

        :param train: The end date of the training series.
        :param val: The end date of the validation series.
        """
        ds = DataSplitting(self.data.get_series())
        ret_split = ds.split(train=train, val=val)
        self._val_exists = val is not None

        self._splits["returns"] = ret_split
        self._splits["indices"] = [s.index for s in ret_split]

    def fit(self, val: bool = False) -> None:
        """
        Trains the hybrid model
        :param val: If true, uses the validation dataset for training as well
        """
        logger.info("Fitting Hybrid Model")
        self._trained_with_val = val
        self._trained = True

        if val and not self._val_exists:
            raise ValueError("Validation split was not created. Call split() first with a valid 'val' parameter.")

        idx_train = self._splits["indices"][0]
        if val:
            idx_train = idx_train.union(self._splits["indices"][1])

        training_featm = self.data.get_feature_matrix().loc[idx_train]
        training_retser = self.data.get_series().loc[idx_train]
        forecasting_training_featm = self.forecasting_data.get_feature_matrix().loc[idx_train]

        self.regime_model = self.regime_model_factory(
            time_series=training_retser,
            feature_matrix=training_featm,
        )
        self.regime_model.fit()
        logger.info("Built Regime Detection Model")
        regime_labels = self.regime_model.get_training_labels()

        if regime_labels.nunique() < 2:
            raise ValueError(f"Only one label present in training data: {regime_labels.unique()}")

        self.forecasting_model = self.forecasting_model_factory(
            feature_matrix=forecasting_training_featm,
            labels=regime_labels
        )
        self.forecasting_model.fit()
        logger.info("Built Forecasting model")

    def predict(self, val: bool = False) -> pd.Series:
        """
        Performs prediction
        :param val: If true then the predictions happen on the validation window
        :return: Labels series
        """
        if not self._trained:
            raise ValueError("Model was not trained. Call fit() first.")

        if val and not self._val_exists:
            raise ValueError("Validation split was not created. Call split() first with a valid 'val' parameter.")

        idx = self._splits["indices"][1] if val else self._splits["indices"][2]
        feat = self.forecasting_data.get_feature_matrix().loc[idx]
        return self.forecasting_model.predict(feat)

    def invest(self, predicted_labels: pd.Series, strategy: Type[BaseStrategy] = LongShortStrategy) -> float:
        """
        Investing based on labels
        :param predicted_labels: The regime labels predicted by a forecasting model
        :param strategy: The investing strategy to use
        :return:
        """
        logger.info("Investing")
        if not self._trained:
            raise ValueError("Model was not trained. Call fit() first.")

        returns = self.data.get_series().loc[predicted_labels.index]

        train_returns = self._splits["returns"][0]
        if self._trained_with_val:
            train_returns = pd.concat([train_returns, self._splits["returns"][1]])
        train_labels = self.regime_model.get_training_labels()

        investor = Investor(
            train_returns=train_returns,
            train_labels=train_labels,
            returns=returns,
        )

        result = investor.invest(
            labels=predicted_labels,
            strategy=strategy,
        )
        self.investment_results = result
        return result["sharpe"]
