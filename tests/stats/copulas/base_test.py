import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from reidfo.stats.copulas import ClaytonCopula


# ---------------------------------------------------------------------------
# TypeError / ValueError guards
# ---------------------------------------------------------------------------

def test_non_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        ClaytonCopula([[1, 2], [3, 4]])


def test_fewer_than_two_columns_raises_value_error():
    index = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame({"A": np.random.randn(50)}, index=index)
    with pytest.raises(ValueError, match="at least two series"):
        ClaytonCopula(df)


def test_non_datetime_index_raises_value_error():
    df = pd.DataFrame({"A": np.random.randn(50), "B": np.random.randn(50)})
    with pytest.raises((ValueError, TypeError)):
        ClaytonCopula(df)


# ---------------------------------------------------------------------------
# fit() required before plot()
# ---------------------------------------------------------------------------

def test_plot_before_fit_raises_runtime_error(price_df):
    cop = ClaytonCopula(price_df)
    with pytest.raises(RuntimeError, match="fit"):
        cop.plot(show=False)


# ---------------------------------------------------------------------------
# _normalize_pairs correctness
# ---------------------------------------------------------------------------

def test_normalize_pairs_none_returns_all(price_df):
    cop = ClaytonCopula(price_df)
    cop.fit()
    result = cop._normalize_pairs(None)
    assert set(result) == set(cop._pairs)


def test_normalize_pairs_single_tuple(price_df):
    cop = ClaytonCopula(price_df)
    cop.fit()
    result = cop._normalize_pairs(("A", "B"))
    assert result == [("A", "B")]


def test_normalize_pairs_single_list(price_df):
    cop = ClaytonCopula(price_df)
    cop.fit()
    result = cop._normalize_pairs(["A", "B"])
    assert result == [("A", "B")]


def test_normalize_pairs_list_of_pairs(price_df):
    cop = ClaytonCopula(price_df)
    cop.fit()
    result = cop._normalize_pairs([("A", "B"), ("A", "C")])
    assert result == [("A", "B"), ("A", "C")]


def test_normalize_pairs_empty_list_returns_all(price_df):
    cop = ClaytonCopula(price_df)
    cop.fit()
    result = cop._normalize_pairs([])
    assert set(result) == set(cop._pairs)
