from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from src.reidfo.core.validation_utils import check_axis_is_date, check_axis_is_string


# reviewed
class ForecastingModel(ABC):
    def __init__(self, feature_matrix: pd.DataFrame, labels: pd.Series, seed: Optional[int]):
        """
        :param feature_matrix: Feature matrix where rows are timestamps and columns are separate time series.
        :param labels: Series of target labels, indexed by the same names as feature columns.
        :param seed: Random seed for reproducibility.
        """
        self.feature_matrix = feature_matrix.copy()
        self.labels = labels.copy()
        self.seed = seed

        check_axis_is_date(self.feature_matrix)
        check_axis_is_string(self.feature_matrix, axis=1)
        if not self.labels.index.equals(self.feature_matrix.index):
            raise ValueError("Labels must be indexed by the same dates as feature matrix columns")

    @abstractmethod
    def fit(self) -> None:
        """
        Train the forecasting model on the stored features and labels.
        """

    @abstractmethod
    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """
        Predict values based on new feature input.

        :param feature_matrix: A DataFrame of features to predict on.
        :return: A Series of predicted regime labels
        """

    @abstractmethod
    def get_model_params(self) -> Optional[dict]:
        """
        Return internal parameters of the trained model, if available.

        :return: A dictionary of model parameters, or None.
        """
