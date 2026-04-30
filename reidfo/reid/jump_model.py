from typing import Literal, Optional

import pandas as pd
from jumpmodels.jump import JumpModel
from numpy.random import RandomState

from .abstract import RegimeModel


class StatisticalJumpModel(RegimeModel):
    def __init__(self,
                 time_series: pd.Series,
                 feature_matrix: pd.DataFrame,
                 n_regimes: int = 2,
                 sort_by: Optional[Literal["cumret", "vol", "freq", "ret", "mean"]] = "cumret",
                 cont: bool = False,
                 prob: bool = False,
                 jump_penalty: float = 0.0,
                 seed: Optional[RandomState | int] = 42):
        """
        Initialize a jump-based regime model.

        :param time_series: Return series aligned with ``feature_matrix``.
        :param feature_matrix: Feature matrix with datetime index and string columns.
        :param n_regimes: Number of regimes.
        :param sort_by: Criterion for sorting regimes. ``"mean"`` triggers a custom
            post-fit sort by per-regime mean of ``time_series``; other values are passed
            straight through to ``JumpModel.fit``.
        :param cont: If True, fits a continuous jump model.
        :param prob: Reserved for probabilistic predictions.
        :param jump_penalty: Penalty controlling regime-switch frequency.
        :param seed: Random state for reproducibility.
        :raises ValueError: If ``feature_matrix`` and ``time_series`` indices differ.
        """
        super().__init__(time_series, feature_matrix, seed)
        self.returns = self.time_series

        if not self.feature_matrix.index.equals(self.returns.index):
            raise ValueError("Index mismatch: 'feature_matrix' and 'time_series' must have identical indices.")

        self.n_regimes = n_regimes
        self.sort_by = sort_by
        self.cont = cont
        self.prob = prob
        self.jump_penalty = jump_penalty

        self.jm: Optional[JumpModel] = None

    def _sort_labels_by_mean(self, labels: pd.Series) -> pd.Series:
        regime_means = self.returns.groupby(labels).mean()
        sorted_regimes = regime_means.sort_values().index
        label_map = {old: new for new, old in enumerate(sorted_regimes)}
        return labels.map(label_map)

    def fit(self) -> None:
        """
        Fit the underlying ``JumpModel`` on the training feature matrix and return series.
        """
        self.jm = JumpModel(
            n_components=self.n_regimes,
            jump_penalty=self.jump_penalty,
            cont=self.cont,
            random_state=self.seed,
        )
        sort_arg = None if self.sort_by == "mean" else self.sort_by
        self.jm.fit(self.feature_matrix, self.returns, sort_by=sort_arg)

        labels = self.jm.labels_
        if self.sort_by == "mean":
            self._labels = self._sort_labels_by_mean(labels)
        else:
            self._labels = labels
        self._fitted = True

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Predict regime labels for new features using ``predict_online``.

        :param feature_matrix: Must share columns with the training matrix and start after the
            last training timestamp.
        :return: Series of predicted regime labels.
        :raises ValueError: If columns or temporal ordering are inconsistent with training.
        """
        super().predict(feature_matrix)
        labels = self.jm.predict_online(feature_matrix)
        if self.sort_by == "mean":
            labels = self._sort_labels_by_mean(labels)
        return labels
