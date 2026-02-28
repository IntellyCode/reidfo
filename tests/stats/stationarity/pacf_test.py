import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf as sm_pacf

from reidfo.stats.stationarity.pacf import PACF


def test_pacf_compute_returns_lag_index_and_series_columns():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, None, None, None, None, None],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "short": [1.0, None, None, None, None, None, None, None, None, None],
        },
        index=index,
    )

    pacf_test = PACF(df)
    scores = pacf_test.compute()

    ts_a = df["a"].dropna().values
    ts_b = df["b"].dropna().values
    expected_a = sm_pacf(ts_a, nlags=int(len(ts_a) * 0.4), method='ywm')
    expected_b = sm_pacf(ts_b, nlags=int(len(ts_b) * 0.4), method='ywm')

    assert list(scores.columns) == ["a", "b", "short"]
    assert list(scores.index) == list(range(max(len(expected_a), len(expected_b))))
    assert scores.index.name == "lag"

    assert np.allclose(scores["a"].dropna().values, expected_a, equal_nan=True)
    assert np.allclose(scores["b"].values, expected_b, equal_nan=True)
    assert np.isnan(scores["short"].values).all()

    assert scores is pacf_test.scores


def test_pacf_compute_caches_result():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    pacf_test = PACF(df)
    first = pacf_test.compute()
    df["a"] = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert pacf_test.compute() is first


def test_pacf_plot_writes_files(tmp_path):
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    PACF(df).plot(str(tmp_path))
    assert (tmp_path / "PACF_a.pdf").is_file()


def test_pacf_plot_uses_cached_data(tmp_path, monkeypatch):
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]},
        index=index,
    )
    pacf_test = PACF(df)
    pacf_test.compute()
    monkeypatch.setattr(PACF, "compute", lambda self: (_ for _ in ()).throw(AssertionError()))
    pacf_test.plot(str(tmp_path))
    assert (tmp_path / "PACF_a.pdf").is_file()
    assert (tmp_path / "PACF_b.pdf").is_file()
