import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from reidfo.stats.general_statistics import GeneralStatistics


def _expected_stats(values: np.ndarray) -> list[float]:
    if len(values) < 2:
        return [np.nan] * 7

    return [
        np.mean(values),
        np.median(values),
        np.std(values, ddof=1),
        np.min(values),
        np.max(values),
        skew(values),
        kurtosis(values),
    ]


def test_general_statistics_compute_columnwise_and_cache():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, np.nan],
            "b": [4.0, np.nan, 6.0, 8.0],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    stats = GeneralStatistics(df)
    first = stats.compute()

    assert list(first.index) == ["a", "b"]
    assert list(first.columns) == ["Mean", "Median", "StdDev", "Min", "Max", "Skew", "Kurt"]

    expected = pd.DataFrame(
        {
            "a": _expected_stats(df["a"].dropna().values),
            "b": _expected_stats(df["b"].dropna().values),
        },
        index=["Mean", "Median", "StdDev", "Min", "Max", "Skew", "Kurt"],
    ).T

    assert np.allclose(first.values, expected.values, equal_nan=True)

    df["a"] = [10.0, 20.0, 30.0, 40.0]
    second = stats.compute()

    assert second is first
    assert np.allclose(second.values, expected.values, equal_nan=True)
