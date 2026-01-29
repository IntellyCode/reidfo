import pandas as pd
import pytest

from reidfo.core.validation_utils import (
    check_columns_are_strings,
    check_df_for_nans,
    check_index_is_datetime,
    validate_minimum_regimes,
)


def test_check_index_is_datetime_accepts_datetime_index():
    df = pd.DataFrame({"a": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))
    check_index_is_datetime(df)


def test_check_index_is_datetime_rejects_non_datetime_index():
    series = pd.Series([1, 2], index=["a", "b"])
    with pytest.raises(ValueError):
        check_index_is_datetime(series)


def test_check_columns_are_strings():
    df = pd.DataFrame({"a": [1], "b": [2]})
    check_columns_are_strings(df)
    df_bad = pd.DataFrame({1: [1], "b": [2]})
    with pytest.raises(ValueError):
        check_columns_are_strings(df_bad)


def test_check_df_for_nans_series_and_df():
    df_ok = pd.DataFrame({"a": [1, 2]})
    check_df_for_nans(df_ok)
    df_bad = pd.DataFrame({"a": [1, None]})
    with pytest.raises(ValueError):
        check_df_for_nans(df_bad)
    series_bad = pd.Series([1, None])
    with pytest.raises(ValueError):
        check_df_for_nans(series_bad)


def test_validate_minimum_regimes():
    labels = pd.Series([0, 1, 0, 1])
    validate_minimum_regimes(labels, required=2)
    df_bad = pd.DataFrame({"a": [0, 1, 2]})
    with pytest.raises(ValueError):
        validate_minimum_regimes(df_bad, required=2)
