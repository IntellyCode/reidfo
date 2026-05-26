import numpy as np
import pandas as pd
import pytest

from reidfo.refo.xgboost import XGBoostModel


def _make_data(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    feat = pd.DataFrame(
        {"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)},
        index=idx,
    )
    labels = pd.Series((rng.standard_normal(n) > 0).astype(int), index=idx)
    return feat, labels


def test_fit_and_predict_returns_series():
    feat, labels = _make_data()
    model = XGBoostModel(feat, labels, seed=0)
    model.fit()

    future_idx = pd.date_range(feat.index[-1] + pd.Timedelta(days=1), periods=10, freq="D")
    rng = np.random.default_rng(1)
    new_feat = pd.DataFrame(
        {"f1": rng.standard_normal(10), "f2": rng.standard_normal(10)},
        index=future_idx,
    )
    preds = model.predict(new_feat)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(new_feat) - 1
    assert preds.index.equals(new_feat.index[1:])


def test_predict_before_fit_raises():
    feat, labels = _make_data()
    model = XGBoostModel(feat, labels, seed=0)
    with pytest.raises(AssertionError):
        model.predict(feat)


def test_get_model_params_after_fit():
    feat, labels = _make_data()
    model = XGBoostModel(feat, labels, seed=0)
    model.fit()
    params = model.get_model_params()
    assert "feature_importance" in params
    assert isinstance(params["feature_importance"], pd.Series)
    assert list(params["feature_importance"].index) == ["f1", "f2"]


def test_invalid_smoothing_halflife_raises():
    feat, labels = _make_data()
    with pytest.raises(ValueError, match="smoothing_halflife"):
        XGBoostModel(feat, labels, smoothing_halflife=-1.0)


def test_label_index_mismatch_raises():
    feat, labels = _make_data()
    misaligned = labels.copy()
    misaligned.index = pd.date_range("2030-01-01", periods=len(labels), freq="D")
    with pytest.raises(ValueError):
        XGBoostModel(feat, misaligned)


def test_non_datetime_index_raises():
    feat, labels = _make_data()
    bad_feat = pd.DataFrame(feat.values, columns=feat.columns, index=range(len(feat)))
    bad_labels = pd.Series(labels.values, index=range(len(labels)))
    with pytest.raises(ValueError):
        XGBoostModel(bad_feat, bad_labels)
