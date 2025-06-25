from typing import Optional, Literal
import pandas as pd
from numpy.random import RandomState
from jumpmodels.jump import JumpModel

from .abstract import RegimeModel


# reviewed
class StatisticalJumpModel(RegimeModel):
    def __init__(self,
                 time_series: pd.Series,
                 feature_matrix: pd.DataFrame,
                 n_regimes: int = 2,
                 sort_by: Optional[Literal["cumret", "vol", "freq", "ret", "mean"]] = 'cumret',
                 cont: bool = False,
                 prob: bool = False,
                 jump_penalty: float = 0.0,

                 seed: Optional[RandomState | int] = 42):
        """
        Initialize a jump-based regime model.

        :param time_series: Corresponding return series with datetime.date index.
        :param feature_matrix: Feature matrix with datetime.date index and string columns.
        :param n_regimes: Number of regimes.
        :param sort_by: Criterion for sorting the states.
        :param cont: If True, the jump model is continuous
        :param prob: If True, returns probabilities instead of discrete labels
        :param seed: Optional random state for reproducibility.
        """
        super().__init__(time_series, feature_matrix, seed)
        self.returns = time_series

        if not self.feature_matrix.index.equals(self.returns.index):
            raise ValueError("Index mismatch: 'feature_matrix' and 'returns' must have identical indices.")

        self.n_regimes = n_regimes
        self.sort_by = sort_by
        self.cont = cont
        self.prob = prob
        self.jump_penalty = jump_penalty
        self.seed = seed

        self.jm: Optional[JumpModel] = None

    def _sort_labels_by_mean(self, labels: pd.Series) -> pd.Series:
        regime_means = self.returns.groupby(labels).mean()
        sorted_regimes = regime_means.sort_values().index
        label_map = {old: new for new, old in enumerate(sorted_regimes)}
        return labels.map(label_map)

    def fit(self) -> None:
        """
        Fits a JumpModel to the training feature matrix and return series.
        """
        X_processed = self.feature_matrix
        self.jm = JumpModel(n_components=self.n_regimes,
                            jump_penalty=self.jump_penalty,
                            cont=self.cont,
                            random_state=self.seed)
        sort_arg = None if self.sort_by == "mean" else self.sort_by
        self.jm.fit(X_processed, self.returns, sort_by=sort_arg)

        labels = self.jm.labels_
        if self.sort_by == "mean":
            self._labels = self._sort_labels_by_mean(labels)
        else:
            self._labels = labels

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Predicts regime labels using the trained JumpModel.

        :param feature_matrix: Must match training matrix columns; index must be future-dated.
        :return: A series of predicted regime labels.
        :raises ValueError: If input format or date alignment is incorrect.
        """
        super().predict(feature_matrix)
        X_processed = self.feature_matrix
        labels = self.jm.predict_online(X_processed)
        if self.sort_by == "mean":
            labels = self._sort_labels_by_mean(labels)
        return labels
