import numpy as np
import pandas as pd
import pytest

from reidfo.reid.abstract import RegimeModel


class _DummyModel(RegimeModel):
    def fit(self) -> None:
        self._fitted = True
        self._labels = pd.Series(0, index=self.feature_matrix.index)


def _make_inputs(n: int = 6, start: str = "2024-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    rng = np.random.default_rng(0)
    feat = pd.DataFrame(rng.standard_normal((n, 2)), index=idx, columns=["f1", "f2"])
    series = pd.Series(rng.standard_normal(n), index=idx)
    return series, feat


def test_init_accepts_aligned_inputs():
    series, feat = _make_inputs()
    model = _DummyModel(series, feat, seed=0)
    assert model.feature_matrix.equals(feat)
    assert model.time_series.equals(series)
    assert model.get_training_labels() is None


def test_init_rejects_non_datetime_index():
    series, feat = _make_inputs()
    bad = feat.reset_index(drop=True)
    with pytest.raises(ValueError):
        _DummyModel(series, bad, seed=0)


def test_init_rejects_non_string_columns():
    series, feat = _make_inputs()
    feat.columns = [1, 2]
    with pytest.raises(ValueError):
        _DummyModel(series, feat, seed=0)


def test_predict_rejects_column_mismatch():
    series, feat = _make_inputs()
    model = _DummyModel(series, feat, seed=0)
    future_idx = pd.date_range(feat.index[-1] + pd.Timedelta(days=1), periods=3, freq="D")
    bad = pd.DataFrame(np.zeros((3, 2)), index=future_idx, columns=["x", "y"])
    with pytest.raises(ValueError, match="same columns"):
        model.predict(bad)


def test_predict_rejects_overlapping_window():
    series, feat = _make_inputs()
    model = _DummyModel(series, feat, seed=0)
    overlap = feat.copy()
    with pytest.raises(ValueError, match="after training data"):
        model.predict(overlap)


def test_get_training_labels_returns_after_fit():
    series, feat = _make_inputs()
    model = _DummyModel(series, feat, seed=0)
    model.fit()
    labels = model.get_training_labels()
    assert isinstance(labels, pd.Series)
    assert labels.index.equals(feat.index)
