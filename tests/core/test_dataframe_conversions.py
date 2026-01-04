import pytest
import pandas as pd
import datetime as dt

from reidfo.core.dataframe_conversions import convert_axis_to_date


class TestConvertAxisToDate:
    def test_convert_index_datetime_to_date(self):
        dates = pd.date_range("2020-01-01", periods=3)
        df = pd.DataFrame({"a": [1, 2, 3]}, index=dates)
        result = convert_axis_to_date(df, axis=0)

        assert all(isinstance(d, dt.date) for d in result.index)
        assert list(result.index) == [dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)]

    def test_convert_columns_datetime_to_date(self):
        dates = pd.date_range("2020-01-01", periods=3)
        df = pd.DataFrame([[1, 2, 3]], columns=dates)
        result = convert_axis_to_date(df, axis=1)

        assert all(isinstance(d, dt.date) for d in result.columns)
        assert list(result.columns) == [dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)]

    def test_convert_string_dates_to_date(self):
        df = pd.DataFrame({"a": [1, 2, 3]}, index=["2020-01-01", "2020-01-02", "2020-01-03"])
        result = convert_axis_to_date(df, axis=0)

        assert all(isinstance(d, dt.date) for d in result.index)
        assert list(result.index) == [dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)]

    def test_convert_series_index_to_date(self):
        dates = pd.date_range("2020-01-01", periods=3)
        series = pd.Series([1, 2, 3], index=dates)
        result = convert_axis_to_date(series, axis=0)

        assert all(isinstance(d, dt.date) for d in result.index)
        assert list(result.index) == [dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)]

    def test_invalid_axis_raises_valueerror(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Axis must be 0 .* or 1"):
            convert_axis_to_date(df, axis=2)

    def test_does_not_mutate_original(self):
        dates = pd.date_range("2020-01-01", periods=3)
        df = pd.DataFrame({"a": [1, 2, 3]}, index=dates)
        original_index = list(df.index)

        result = convert_axis_to_date(df, axis=0)

        assert list(df.index) == original_index
        assert result is not df

    def test_returns_correct_type(self):
        dates = pd.date_range("2020-01-01", periods=3)
        df = pd.DataFrame({"a": [1, 2, 3]}, index=dates)
        series = pd.Series([1, 2, 3], index=dates)

        df_result = convert_axis_to_date(df, axis=0)
        series_result = convert_axis_to_date(series, axis=0)

        assert isinstance(df_result, pd.DataFrame)
        assert isinstance(series_result, pd.Series)

    def test_preserves_data_values(self):
        dates = pd.date_range("2020-01-01", periods=3)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=dates)
        result = convert_axis_to_date(df, axis=0)

        assert list(result["a"]) == [1, 2, 3]
        assert list(result["b"]) == [4, 5, 6]