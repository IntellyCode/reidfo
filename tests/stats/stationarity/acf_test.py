import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf as sm_acf

from reidfo.stats.stationarity.acf import ACF


def test_acf_compute_returns_lag_index_and_series_columns():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "short": [1, None, None, None, None],
        },
        index=index,
    )

    acf_test = ACF(df)
    scores = acf_test.compute()

    assert list(scores.columns) == ["a", "b", "short"]
    assert list(scores.index) == list(range(5))
    assert scores.index.name == "lag"

    expected_a = sm_acf(df["a"].dropna().values, nlags=4, fft=True)
    expected_b = sm_acf(df["b"].dropna().values, nlags=4, fft=True)

    assert np.allclose(scores["a"].values, expected_a, equal_nan=True)
    assert np.allclose(scores["b"].values, expected_b, equal_nan=True)
    assert np.isnan(scores["short"].values).all()
    assert scores is acf_test.scores


def test_acf_plot_writes_files(tmp_path):
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=index)

    acf_test = ACF(df)
    acf_test.plot(str(tmp_path))

    assert (tmp_path / "ACF_a.png").is_file()
