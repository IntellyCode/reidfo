from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger
from xgboost import XGBClassifier

from .forecasting_model import ForecastingModel


class XGBoostModel(ForecastingModel):
    def __init__(self,
                 feature_matrix: pd.DataFrame,
                 labels: pd.Series,
                 hyperparams: Optional[Dict[str, Any]] = None,
                 smoothing_halflife: Optional[float] = 0.1,
                 seed: int = 42):
        """
        XGBoost-based forecasting model for discrete labels (e.g. regimes).

        :param feature_matrix: DataFrame of training features.
        :param labels: Series of training labels.
        :param hyperparams: Optional dict of XGBoost hyperparameters.
        :param smoothing_halflife: EWM half-life applied to predicted probabilities;
            ``None`` disables smoothing.
        :param seed: Random seed for reproducibility.
        :raises ValueError: If ``smoothing_halflife`` is non-positive.
        """
        super().__init__(feature_matrix, labels, seed)
        if smoothing_halflife is not None and smoothing_halflife <= 0:
            raise ValueError("smoothing_halflife must be positive if set.")

        params: Dict[str, Any] = {
            "random_state": self.seed,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "n_jobs": 1,
        }
        if hyperparams:
            params.update(hyperparams)

        self.model = XGBClassifier(**params)
        self._trained = False
        self._feature_importance: Optional[pd.Series] = None
        self._smoothing_halflife = smoothing_halflife
        self._forecasted_probabilities: list = []

    def fit(self) -> None:
        """
        Fit the model to predict ``label[t+1]`` using features at time ``t``.
        """
        X = self.feature_matrix.values[:-1]
        y = self.labels.values[1:]

        self.model.fit(X, y)
        self._trained = True
        self._feature_importance = pd.Series(
            self.model.feature_importances_, index=self.feature_matrix.columns
        )

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """
        Predict future labels using the trained model with optional EWM smoothing.

        :param feature_matrix: DataFrame of features to predict on.
        :return: Series of predicted class labels, indexed from ``feature_matrix.index[1:]``.
        :raises AssertionError: If the model has not been trained yet.
        """
        assert self._trained, "Model not trained yet!"
        probs = self.model.predict_proba(feature_matrix.values)
        self._forecasted_probabilities.extend(probs.tolist())
        logger.debug(f"Probabilities: {self._forecasted_probabilities}")
        prob_df = pd.DataFrame(self._forecasted_probabilities)
        if self._smoothing_halflife is not None:
            prob_df = prob_df.ewm(halflife=self._smoothing_halflife).mean()
        self._forecasted_probabilities = prob_df.values.tolist()
        n_rows = feature_matrix.shape[0]
        preds = prob_df.iloc[-n_rows:, :].values.argmax(axis=1)
        return pd.Series(preds[:-1], index=feature_matrix.index[1:])

    def get_model_params(self) -> Optional[dict]:
        """
        Return the fitted model parameters.

        :return: Dict with ``feature_importance``, ``model_config``, and ``smoothing_halflife``.
        """
        return {
            "feature_importance": self._feature_importance,
            "model_config": self.model.get_params(),
            "smoothing_halflife": self._smoothing_halflife,
        }
