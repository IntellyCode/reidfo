import pytest
import pandas as pd
import numpy as np
import datetime as dt

from reidfo.core.data_splitting import DataSplitting


class TestDataSplittingInit:
    def test_init_with_series(self):
        series = pd.Series([1, 2, 3, 4, 5])
        ds = DataSplitting(series)
        assert isinstance(ds.data, pd.Series)
        assert len(ds.data) == 5

    def test_init_with_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds = DataSplitting(df)
        assert isinstance(ds.data, pd.DataFrame)

    def test_init_makes_copy(self):
        series = pd.Series([1, 2, 3, 4, 5])
        ds = DataSplitting(series)
        ds.data.iloc[0] = 999
        assert series.iloc[0] == 1

    def test_init_with_nans_raises_valueerror(self):
        series = pd.Series([1, np.nan, 3])
        with pytest.raises(ValueError, match="Input data contains NaNs"):
            DataSplitting(series)


class TestDataSplittingSplitByProportion:
    def test_split_series_train_only(self):
        series = pd.Series(range(10))
        ds = DataSplitting(series)
        result = ds.split(train=0.6)

        assert len(result) == 3
        assert len(result[0]) == 6  # train
        assert len(result[1]) == 0  # val
        assert len(result[2]) == 4  # test

    def test_split_series_train_and_val(self):
        series = pd.Series(range(10))
        ds = DataSplitting(series)
        result = ds.split(train=0.5, val=0.3)

        assert len(result) == 3
        assert len(result[0]) == 5  # train
        assert len(result[1]) == 3  # val
        assert len(result[2]) == 2  # test

    def test_split_dataframe_by_proportion(self):
        df = pd.DataFrame({
            i: [i, i * 2] for i in range(10)
        }, index=["row1", "row2"])
        ds = DataSplitting(df)
        result = ds.split(train=0.5, val=0.3)

        assert isinstance(result, dict)
        assert "row1" in result
        assert "row2" in result
        assert len(result["row1"]) == 3
        assert len(result["row1"][0]) == 5

    def test_split_proportions_exceed_one_raises_valueerror(self):
        series = pd.Series(range(10))
        ds = DataSplitting(series)
        with pytest.raises(ValueError, match="Train and val proportions exceed 1.0"):
            ds.split(train=0.6, val=0.5)

    def test_split_allows_no_test(self):
        series = pd.Series(range(10))
        ds = DataSplitting(series)
        result = ds.split(train=0.8, val=0.2)

        assert len(result[0]) == 8  # train
        assert len(result[1]) == 2  # val
        assert len(result[2]) == 0  # test (empty)

    def test_split_correct_data_ordering(self):
        series = pd.Series(range(10))
        ds = DataSplitting(series)
        result = ds.split(train=0.5, val=0.3)

        assert list(result[0]) == [0, 1, 2, 3, 4]
        assert list(result[1]) == [5, 6, 7]
        assert list(result[2]) == [8, 9]


class TestDataSplittingSplitByDate:
    def test_split_series_by_date(self):
        dates = [dt.date(2020, 1, i) for i in range(1, 11)]
        series = pd.Series(range(10), index=dates)
        ds = DataSplitting(series)

        result = ds.split(train=dt.date(2020, 1, 5), val=dt.date(2020, 1, 8))

        assert len(result) == 3
        assert len(result[0]) == 4  # train: days 1-4
        assert len(result[1]) == 3  # val: days 5-7
        assert len(result[2]) == 3  # test: days 8-10

    def test_split_series_by_date_no_val(self):
        dates = [dt.date(2020, 1, i) for i in range(1, 11)]
        series = pd.Series(range(10), index=dates)
        ds = DataSplitting(series)

        result = ds.split(train=dt.date(2020, 1, 5))

        assert len(result[0]) == 4  # train
        assert len(result[1]) == 0  # val (empty)
        assert len(result[2]) == 6  # test

    def test_split_dataframe_by_date(self):
        dates = [dt.date(2020, 1, i) for i in range(1, 11)]
        df = pd.DataFrame({
            d: [i, i * 2] for i, d in enumerate(dates)
        }, index=["row1", "row2"])
        ds = DataSplitting(df)

        result = ds.split(train=dt.date(2020, 1, 5), val=dt.date(2020, 1, 8))

        assert isinstance(result, dict)
        assert "row1" in result
        assert len(result["row1"]) == 3

    def test_split_date_not_in_index_raises_valueerror(self):
        dates = [dt.date(2020, 1, i) for i in range(1, 11)]
        series = pd.Series(range(10), index=dates)
        ds = DataSplitting(series)

        with pytest.raises(ValueError):
            ds.split(train=dt.date(2020, 2, 1))

    def test_split_dates_wrong_order_raises_valueerror(self):
        dates = [dt.date(2020, 1, i) for i in range(1, 11)]
        series = pd.Series(range(10), index=dates)
        ds = DataSplitting(series)

        with pytest.raises(ValueError, match="Expected date order"):
            ds.split(train=dt.date(2020, 1, 8), val=dt.date(2020, 1, 5))


class TestDataSplittingTypeValidation:
    def test_mixed_types_raises_typeerror(self):
        series = pd.Series(range(10))
        ds = DataSplitting(series)

        with pytest.raises(TypeError, match="must be of the same type"):
            ds.split(train=0.5, val=dt.date(2020, 1, 5))

    def test_invalid_type_raises_typeerror(self):
        series = pd.Series(range(10))
        ds = DataSplitting(series)

        with pytest.raises(TypeError, match="must all be floats or datetime.date"):
            ds.split(train="invalid")


class TestSliceByProportion:
    def test_slice_by_proportion_static(self):
        series = pd.Series(range(10))
        result = DataSplitting._slice_by_proportion(series, 0.5, 0.3)

        assert len(result) == 3
        assert list(result[0]) == [0, 1, 2, 3, 4]
        assert list(result[1]) == [5, 6, 7]
        assert list(result[2]) == [8, 9]


class TestSliceByDate:
    def test_slice_by_date_static(self):
        dates = [dt.date(2020, 1, i) for i in range(1, 11)]
        series = pd.Series(range(10), index=dates)
        result = DataSplitting._slice_by_date(series, dt.date(2020, 1, 5), dt.date(2020, 1, 8))

        assert len(result) == 3
        assert len(result[0]) == 4
        assert len(result[1]) == 3
        assert len(result[2]) == 3