import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from reidfo.feature_engineering.feature_engineer import FeatureEngineer
from reidfo.feature_engineering.collector.base_collector import BaseCollector
from reidfo.feature_engineering.time_series_data import TimeSeriesData


class DummyCollector(BaseCollector):
    def __init__(self):
        super().__init__({})

    def collect(self, time_series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "level": time_series,
                "lagged": time_series.shift(1).fillna(-1.0),
            },
            index=time_series.index,
        )


class NanCollector(BaseCollector):
    def __init__(self):
        super().__init__({})

    def collect(self, time_series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "level": time_series,
                "nan_feature": pd.Series([0.0, float("nan"), 1.0], index=time_series.index),
            },
            index=time_series.index,
        )


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [100.0, 110.0, 121.0, 133.1],
            "b": [50.0, 55.0, 49.5, 54.45],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )


def test_get_data_returns_timeseries_data_with_return_series(df):
    engineer = FeatureEngineer(df, clipper=None, scaler=None)

    result = engineer.get_data("a", DummyCollector())

    expected_series = df["a"].pct_change(fill_method=None).iloc[1:]
    expected_features = pd.DataFrame(
        {
            "level": expected_series,
            "lagged": [-1.0, 0.1, 0.1],
        },
        index=expected_series.index,
    )

    assert isinstance(result, TimeSeriesData)
    assert_series_equal(result.series, expected_series)
    assert_frame_equal(result.feature_matrix, expected_features)


def test_get_data_original_uses_original_series(df):
    engineer = FeatureEngineer(df, clipper=None, scaler=None)

    result = engineer.get_data("b", DummyCollector(), original=True)

    expected_series = df["b"]
    expected_features = pd.DataFrame(
        {
            "level": expected_series,
            "lagged": [-1.0, 50.0, 55.0, 49.5],
        },
        index=expected_series.index,
    )

    assert_series_equal(result.series, expected_series)
    assert_frame_equal(result.feature_matrix, expected_features)


def test_get_data_filters_series_and_features_by_date(df):
    engineer = FeatureEngineer(df, clipper=None, scaler=None)
    start_date = df.index[2]
    end_date = df.index[3]

    result = engineer.get_data("a", DummyCollector(), start_date=start_date, end_date=end_date)

    expected_index = list(df.index[2:4])
    assert list(result.series.index) == expected_index
    assert list(result.feature_matrix.index) == expected_index


def test_get_data_applies_clipper_then_scaler(df):
    engineer = FeatureEngineer(
        df,
        clipper=lambda featm: featm + 1.0,
        scaler=lambda featm: featm * 2.0,
    )

    result = engineer.get_data("a", DummyCollector())

    expected_series = df["a"].pct_change(fill_method=None).iloc[1:]
    expected_features = pd.DataFrame(
        {
            "level": (expected_series + 1.0) * 2.0,
            "lagged": pd.Series([-1.0, 0.1, 0.1], index=expected_series.index).add(1.0).mul(2.0),
        },
        index=expected_series.index,
    )

    assert_frame_equal(result.feature_matrix, expected_features)


def test_get_data_rejects_unknown_column(df):
    engineer = FeatureEngineer(df, clipper=None, scaler=None)

    with pytest.raises(ValueError, match="Column 'missing' not found."):
        engineer.get_data("missing", DummyCollector())


def test_get_data_rejects_nan_feature_matrix(df):
    engineer = FeatureEngineer(df, clipper=None, scaler=None)

    with pytest.raises(ValueError, match="Feature matrix contains NaNs."):
        engineer.get_data("a", NanCollector())
