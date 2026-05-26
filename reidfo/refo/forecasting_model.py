from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from reidfo.core.validation_utils import check_index_is_datetime, check_columns_are_strings


class ForecastingModel(ABC):
    def __init__(self, feature_matrix: pd.DataFrame, labels: pd.Series, seed: Optional[int]):
        """
        Abstract base class for supervised regime forecasting models.

        :param feature_matrix: DataFrame with a datetime index and string column names;
            rows are timestamps, columns are separate time series.
        :param labels: Target labels indexed by the same datetime index as ``feature_matrix``.
        :param seed: Random seed for reproducibility.
        :raises ValueError: If index is not datetime, columns are not strings, or index alignment fails.
        """
        self.feature_matrix = feature_matrix.copy()
        self.labels = labels.copy()
        self.seed = seed

        check_index_is_datetime(self.feature_matrix)
        check_columns_are_strings(self.feature_matrix)
        if not self.labels.index.equals(self.feature_matrix.index):
            raise ValueError("Labels must be indexed by the same dates as the feature matrix.")

    @abstractmethod
    def fit(self) -> None:
        """
        Train the forecasting model on the stored features and labels.
        """

    @abstractmethod
    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """
        Predict values based on new feature input.

        :param feature_matrix: DataFrame of features to predict on.
        :return: Series of predicted regime labels.
        """

    @abstractmethod
    def get_model_params(self) -> Optional[dict]:
        """
        Return internal parameters of the trained model, if available.

        :return: Dictionary of model parameters, or ``None``.
        """
