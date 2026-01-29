import pandas as pd

from reidfo.core.dataframe_conversions import convert_index_to_datetime


def test_convert_index_to_datetime_dataframe():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["2020-01-01", "2020-01-02"])

    result = convert_index_to_datetime(df)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert list(result.index) == list(pd.to_datetime(df.index))
    expected = df.copy()
    expected.index = pd.to_datetime(df.index)
    assert result.equals(expected)


def test_convert_index_to_datetime_series():
    series = pd.Series([1, 2], index=["2021-02-01", "2021-02-02"], name="values")

    result = convert_index_to_datetime(series)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert list(result.index) == list(pd.to_datetime(series.index))
    expected = series.copy()
    expected.index = pd.to_datetime(series.index)
    assert result.equals(expected)
