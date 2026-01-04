import pytest
import pandas as pd
import numpy as np
import datetime as dt

from reidfo.core.validation_utils import (
    check_axis_is_date,
    check_axis_is_string,
    check_df_for_nans,
    validate_minimum_regimes,
)


class TestCheckAxisIsDate:
    def test_valid_date_index_series(self):
        dates = pd.date_range("2020-01-01", periods=5)
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        check_axis_is_date(series, axis=0)

    def test_valid_date_index_dataframe(self):
        dates = pd.date_range("2020-01-01", periods=5)
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=dates)
        check_axis_is_date(df, axis=0)

    def test_valid_date_columns_dataframe(self):
        dates = pd.date_range("2020-01-01", periods=3)
        df = pd.DataFrame([[1, 2, 3]], columns=dates)
        check_axis_is_date(df, axis=1)

    def test_valid_string_date_index(self):
        series = pd.Series([1, 2, 3], index=["2020-01-01", "2020-01-02", "2020-01-03"])
        check_axis_is_date(series, axis=0)

    def test_invalid_index_raises_valueerror(self):
        series = pd.Series([1, 2, 3], index=["a", "b", "c"])
        with pytest.raises(ValueError, match="Index must be convertible to datetime.date"):
            check_axis_is_date(series, axis=0)

    def test_invalid_columns_raises_valueerror(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        with pytest.raises(ValueError, match="Columns must be convertible to datetime.date"):
            check_axis_is_date(df, axis=1)


class TestCheckAxisIsString:
    def test_valid_string_index_series(self):
        series = pd.Series([1, 2, 3], index=["a", "b", "c"])
        check_axis_is_string(series, axis=0)

    def test_valid_string_index_dataframe(self):
        df = pd.DataFrame({"col": [1, 2, 3]}, index=["a", "b", "c"])
        check_axis_is_string(df, axis=0)

    def test_valid_string_columns_dataframe(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        check_axis_is_string(df, axis=1)

    def test_integer_index_raises_valueerror(self):
        series = pd.Series([1, 2, 3], index=[0, 1, 2])
        with pytest.raises(ValueError, match="All column names must be strings"):
            check_axis_is_string(series, axis=0)

    def test_date_index_raises_valueerror(self):
        dates = pd.date_range("2020-01-01", periods=3)
        series = pd.Series([1, 2, 3], index=dates)
        with pytest.raises(ValueError, match="All column names must be strings"):
            check_axis_is_string(series, axis=0)


class TestCheckDfForNans:
    def test_series_without_nans(self):
        series = pd.Series([1, 2, 3, 4, 5])
        check_df_for_nans(series)

    def test_dataframe_without_nans(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        check_df_for_nans(df)

    def test_series_with_nans_raises_valueerror(self):
        series = pd.Series([1, np.nan, 3])
        with pytest.raises(ValueError, match="Input data contains NaNs"):
            check_df_for_nans(series)

    def test_dataframe_with_nans_raises_valueerror(self):
        df = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="Input data contains NaNs"):
            check_df_for_nans(df)

    def test_empty_series(self):
        series = pd.Series([], dtype=float)
        check_df_for_nans(series)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        check_df_for_nans(df)


class TestValidateMinimumRegimes:
    def test_series_with_exactly_two_regimes(self):
        labels = pd.Series([0, 0, 1, 1, 0, 1])
        validate_minimum_regimes(labels, required=2)

    def test_series_with_three_regimes(self):
        labels = pd.Series([0, 1, 2, 0, 1, 2])
        validate_minimum_regimes(labels, required=3)

    def test_series_with_wrong_number_raises_valueerror(self):
        labels = pd.Series([0, 0, 0, 0])
        with pytest.raises(ValueError, match="Exactly 2 regimes required; found 1"):
            validate_minimum_regimes(labels, required=2)

    def test_series_with_nans_ignored(self):
        labels = pd.Series([0, 1, np.nan, 0, 1])
        validate_minimum_regimes(labels, required=2)

    def test_dataframe_all_rows_valid(self):
        df = pd.DataFrame({
            "a": [0, 0],
            "b": [1, 1],
            "c": [0, 1],
        })
        validate_minimum_regimes(df, required=2)

    def test_dataframe_invalid_row_raises_valueerror(self):
        df = pd.DataFrame({
            "a": [0, 0],
            "b": [1, 0],
            "c": [0, 0],
        })
        with pytest.raises(ValueError, match="Each row must contain exactly 2 regimes"):
            validate_minimum_regimes(df, required=2)

    def test_invalid_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="Labels must be a pandas Series or DataFrame"):
            validate_minimum_regimes([0, 1, 0, 1], required=2)

    def test_custom_required_count(self):
        labels = pd.Series([0, 1, 2, 3, 0, 1, 2, 3])
        validate_minimum_regimes(labels, required=4)