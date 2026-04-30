from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from reidfo.core.validation_utils import check_columns_are_strings, check_index_is_datetime


class RegimeModel(ABC):
    def __init__(self, time_series: pd.Series, feature_matrix: pd.DataFrame, seed: Optional[int]):
        """
        Abstract base class for regime models.

        Stores feature matrix and aligned time series for supervised regime labelling.
        Verifies index and column conventions on construction.

        :param time_series: Series indexed by datetime.
        :param feature_matrix: DataFrame with a datetime index and string column names.
        :param seed: Random seed for reproducibility.
        :raises ValueError: If index is not datetime or columns are not strings.
        """
        self.time_series = time_series.copy()
        self.feature_matrix = feature_matrix.copy()
        self.seed = seed
        self._labels: Optional[pd.Series] = None
        self._fitted = False

        check_index_is_datetime(self.feature_matrix)
        check_index_is_datetime(self.time_series)
        check_columns_are_strings(self.feature_matrix)

    @abstractmethod
    def fit(self) -> None:
        """
        Fit the model to the training data.
        """
        self._fitted = True

    def predict(self, feature_matrix: pd.DataFrame) -> Optional[pd.Series | pd.DataFrame]:
        """
        Predict regimes from new features. Subclasses override and call ``super().predict``
        for shared validation. Ensures the prediction matrix matches the training schema
        and lies strictly after the training window.

        :param feature_matrix: Feature matrix with the same columns as training and a future-dated index.
        :return: ``None`` from the base class; subclasses return predicted labels.
        :raises ValueError: If columns mismatch or the prediction window precedes training.
        """
        if set(feature_matrix.columns) != set(self.feature_matrix.columns):
            raise ValueError(
                "Prediction feature matrix must have the same columns as training matrix.\n"
                f"Feature matrix: {feature_matrix.columns}\n"
                f"Training matrix: {self.feature_matrix.columns}\n"
            )
        if feature_matrix.index[0] < self.feature_matrix.index[-1]:
            raise ValueError("Prediction data must start after training data (time series order).")
        return None

    def get_training_labels(self) -> Optional[pd.Series]:
        """
        Return the regime labels generated during training.

        :return: Series of training labels, or ``None`` if the model was not fit.
        """
        return self._labels
