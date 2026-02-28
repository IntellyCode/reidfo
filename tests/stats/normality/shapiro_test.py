import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.stats import shapiro as sp_shapiro

from reidfo.stats.normality.shapiro import ShapiroWilkTest


def test_shapiro_compute_returns_series_index_and_expected_columns():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "a": [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0],
            "b": list(range(1, 11)),
            "short": [1.0, 2.0, None, None, None, None, None, None, None, None],
        },
        index=index,
    )

    sw_test = ShapiroWilkTest(df)
    scores = sw_test.compute()

    assert list(scores.index) == ["a", "b", "short"]
    assert list(scores.columns) == ["Shapiro-Wilk", "p-value", "normal"]

    ts_a = df["a"].dropna().values
    ts_b = df["b"].dropna().values
    res_a = sp_shapiro(ts_a)
    res_b = sp_shapiro(ts_b)

    assert np.isclose(scores.loc["a", "Shapiro-Wilk"], res_a.statistic)
    assert np.isclose(scores.loc["a", "p-value"], res_a.pvalue)
    assert scores.loc["a", "normal"] == (res_a.pvalue > 0.05)

    assert np.isclose(scores.loc["b", "Shapiro-Wilk"], res_b.statistic)
    assert np.isclose(scores.loc["b", "p-value"], res_b.pvalue)
    assert scores.loc["b", "normal"] == (res_b.pvalue > 0.05)

    assert np.isnan(scores.loc["short", "Shapiro-Wilk"])
    assert np.isnan(scores.loc["short", "p-value"])
    assert np.isnan(scores.loc["short", "normal"])

    assert scores is sw_test.scores


def test_shapiro_compute_caches_result():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"a": [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0]},
        index=index,
    )
    sw_test = ShapiroWilkTest(df)
    first = sw_test.compute()
    df["a"] = [10.0] * 10
    assert sw_test.compute() is first
