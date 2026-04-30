import numpy as np
import pandas as pd
import pytest

from reidfo.reid.regime_stats import RegimeStats


def _series_inputs():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    series = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03], index=idx)
    labels = pd.Series([0, 1, 0, 1, 0, 1], index=idx)
    return series, labels


def test_series_input_returns_per_regime_aggregates():
    series, labels = _series_inputs()
    stats = RegimeStats(series, labels, returns=True).get_regime_stats()
    assert set(stats.columns) == {"mean", "std", "count", "cumret", "scaled_cumret"}
    assert list(stats.index) == [0, 1]
    assert stats.loc[0, "count"] == 3
    assert stats.loc[1, "count"] == 3
    np.testing.assert_allclose(stats.loc[0, "mean"], series[labels == 0].mean())
    np.testing.assert_allclose(stats.loc[1, "cumret"], (1 + series[labels == 1]).prod() - 1)


def test_series_input_without_returns_omits_return_columns():
    series, labels = _series_inputs()
    stats = RegimeStats(series, labels, returns=False).get_regime_stats()
    assert set(stats.columns) == {"mean", "std", "count"}


def test_series_input_rejects_index_mismatch():
    series, labels = _series_inputs()
    misaligned = labels.copy()
    misaligned.index = pd.date_range("2030-01-01", periods=len(misaligned), freq="D")
    with pytest.raises(ValueError, match="Index mismatch"):
        RegimeStats(series, misaligned)


def test_series_input_rejects_non_datetime_index():
    series, labels = _series_inputs()
    bad = pd.Series(series.to_numpy(), index=range(len(series)))
    bad_labels = pd.Series(labels.to_numpy(), index=range(len(labels)))
    with pytest.raises(ValueError):
        RegimeStats(bad, bad_labels)


def test_dataframe_input_aggregates_each_row():
    cols = pd.date_range("2024-01-01", periods=4, freq="D")
    ts = pd.DataFrame(
        [[0.01, -0.02, 0.03, -0.04], [0.05, 0.06, -0.07, 0.08]],
        index=["A", "B"],
        columns=cols,
    )
    lbl = pd.DataFrame(
        [[0, 1, 0, 1], [1, 1, 0, 0]],
        index=["A", "B"],
        columns=cols,
    )
    stats = RegimeStats(ts, lbl, returns=False).get_regime_stats()
    assert set(stats.columns.get_level_values(0)) == {"A", "B"}
    np.testing.assert_allclose(stats[("A",)].loc[0, "mean"], np.mean([0.01, 0.03]))
    np.testing.assert_allclose(stats[("B",)].loc[1, "count"], 2)


def test_dataframe_rejects_mismatched_columns():
    cols = pd.date_range("2024-01-01", periods=3, freq="D")
    other = pd.date_range("2025-01-01", periods=3, freq="D")
    ts = pd.DataFrame(np.zeros((1, 3)), index=["A"], columns=cols)
    lbl = pd.DataFrame(np.zeros((1, 3), dtype=int), index=["A"], columns=other)
    with pytest.raises(ValueError, match="Column mismatch"):
        RegimeStats(ts, lbl)


def test_mixed_types_raise_type_error():
    series, labels = _series_inputs()
    with pytest.raises(TypeError):
        RegimeStats(series, labels.to_frame())
