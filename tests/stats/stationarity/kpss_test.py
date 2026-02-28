import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import kpss as sm_kpss

from reidfo.stats.stationarity.kpss import KPSS


def test_kpss_compute_returns_series_index_and_expected_columns():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "a": [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0],
            "b": list(range(1, 11)),
            "short": [1.0, 2.0, 3.0, None, None, None, None, None, None, None],
        },
        index=index,
    )

    kpss_test = KPSS(df)
    scores = kpss_test.compute()

    assert list(scores.index) == ["a", "b", "short"]
    assert list(scores.columns) == ["KPSS_stat", "p_value", "stationary"]

    ts_a = df["a"].dropna().values
    ts_b = df["b"].dropna().values
    expected_stat_a, expected_pval_a, _, _ = sm_kpss(ts_a, regression='c', nlags='auto')
    expected_stat_b, expected_pval_b, _, _ = sm_kpss(ts_b, regression='c', nlags='auto')

    assert np.isclose(scores.loc["a", "KPSS_stat"], expected_stat_a)
    assert np.isclose(scores.loc["a", "p_value"], expected_pval_a)
    assert scores.loc["a", "stationary"] == (expected_pval_a > 0.05)

    assert np.isclose(scores.loc["b", "KPSS_stat"], expected_stat_b)
    assert np.isclose(scores.loc["b", "p_value"], expected_pval_b)
    assert scores.loc["b", "stationary"] == (expected_pval_b > 0.05)

    assert np.isnan(scores.loc["short", "KPSS_stat"])
    assert np.isnan(scores.loc["short", "p_value"])
    assert np.isnan(scores.loc["short", "stationary"])

    assert scores is kpss_test.scores


def test_kpss_compute_caches_result():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"a": [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0]},
        index=index,
    )
    kpss_test = KPSS(df)
    first = kpss_test.compute()
    df["a"] = [10.0] * 10
    assert kpss_test.compute() is first


def test_kpss_plot_is_noop(tmp_path):
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"a": [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0]},
        index=index,
    )
    KPSS(df).plot(str(tmp_path))
    assert list(tmp_path.iterdir()) == []
