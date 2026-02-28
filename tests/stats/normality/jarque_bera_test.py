import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera as sp_jarque_bera

from reidfo.stats.normality.jarque_bera import JarqueBeraTest


def test_jarque_bera_compute_returns_series_index_and_expected_columns():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "a": [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0],
            "b": list(range(1, 11)),
            "short": [1.0, 2.0, None, None, None, None, None, None, None, None],
        },
        index=index,
    )

    jb_test = JarqueBeraTest(df)
    scores = jb_test.compute()

    assert list(scores.index) == ["a", "b", "short"]
    assert list(scores.columns) == ["Jarque-Bera", "p-value", "normal"]

    ts_a = df["a"].dropna().values
    ts_b = df["b"].dropna().values
    res_a = sp_jarque_bera(ts_a)
    res_b = sp_jarque_bera(ts_b)

    assert np.isclose(scores.loc["a", "Jarque-Bera"], res_a.statistic)
    assert np.isclose(scores.loc["a", "p-value"], res_a.pvalue)
    assert scores.loc["a", "normal"] == (res_a.pvalue > 0.05)

    assert np.isclose(scores.loc["b", "Jarque-Bera"], res_b.statistic)
    assert np.isclose(scores.loc["b", "p-value"], res_b.pvalue)
    assert scores.loc["b", "normal"] == (res_b.pvalue > 0.05)

    assert np.isnan(scores.loc["short", "Jarque-Bera"])
    assert np.isnan(scores.loc["short", "p-value"])
    assert np.isnan(scores.loc["short", "normal"])

    assert scores is jb_test.scores


def test_jarque_bera_compute_caches_result():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"a": [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.2, 1.1, 0.9, 1.0]},
        index=index,
    )
    jb_test = JarqueBeraTest(df)
    first = jb_test.compute()
    df["a"] = [10.0] * 10
    assert jb_test.compute() is first
