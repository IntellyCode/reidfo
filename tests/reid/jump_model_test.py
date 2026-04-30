import numpy as np
import pandas as pd
import pytest

from reidfo.reid.jump_model import StatisticalJumpModel


def _make_two_regime_data(n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    half = n // 2
    returns = np.concatenate([rng.normal(-0.05, 0.005, half), rng.normal(0.05, 0.005, n - half)])
    feat = pd.DataFrame(
        {
            "ret": returns,
            "absret": np.abs(returns),
        },
        index=idx,
    )
    series = pd.Series(returns, index=idx)
    return series, feat


def test_fit_produces_two_regime_labels_with_default_sort():
    series, feat = _make_two_regime_data()
    model = StatisticalJumpModel(series, feat, n_regimes=2, jump_penalty=0.0, seed=42)
    model.fit()
    labels = model.get_training_labels()
    assert isinstance(labels, pd.Series)
    assert set(labels.unique()) <= {0, 1}
    assert labels.nunique() == 2


def test_predict_uses_provided_feature_matrix_not_training():
    """Regression guard: predict must operate on the new feature_matrix argument,
    not silently fall back to the training matrix."""
    series, feat = _make_two_regime_data()
    model = StatisticalJumpModel(series, feat, n_regimes=2, jump_penalty=0.0, seed=42)
    model.fit()

    rng = np.random.default_rng(1)
    future_idx = pd.date_range(feat.index[-1] + pd.Timedelta(days=1), periods=20, freq="D")
    new_returns = rng.normal(0.05, 0.005, 20)
    new_feat = pd.DataFrame({"ret": new_returns, "absret": np.abs(new_returns)}, index=future_idx)

    preds = model.predict(new_feat)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(new_feat)
    assert preds.index.equals(new_feat.index)


def test_sort_by_mean_orders_regimes_by_return_mean():
    series, feat = _make_two_regime_data()
    model = StatisticalJumpModel(series, feat, n_regimes=2, sort_by="mean", jump_penalty=0.0, seed=42)
    model.fit()
    labels = model.get_training_labels()
    means = series.groupby(labels).mean().sort_index()
    assert means.is_monotonic_increasing


def test_index_mismatch_raises():
    series, feat = _make_two_regime_data()
    misaligned = series.iloc[1:]
    with pytest.raises(ValueError, match="Index mismatch"):
        StatisticalJumpModel(misaligned, feat, n_regimes=2, seed=42)
