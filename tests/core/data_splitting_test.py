import pandas as pd
import pytest

from reidfo.core.data_splitting import DataSplitting


@pytest.fixture()
def default_df() -> pd.DataFrame:
    index = pd.date_range("1996-01-01", "2006-12-01", freq="MS")
    data = {
        "Austria": range(len(index)),
        "Germany": range(1000, 1000 + len(index)),
        "Switzerland": range(2000, 2000 + len(index)),
    }
    return pd.DataFrame(data, index=index)


def test_to_frame_dataframe_returns_copy(default_df: pd.DataFrame) -> None:
    result = DataSplitting._to_frame(default_df)

    assert result is not default_df
    assert result.equals(default_df)


def test_to_frame_series_to_dataframe(default_df: pd.DataFrame) -> None:
    series = default_df["Austria"].copy()

    result = DataSplitting._to_frame(series)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["series"]
    expected = series.copy()
    expected.name = "series"
    pd.testing.assert_series_equal(result["series"], expected)


def test_init_converts_series(default_df: pd.DataFrame) -> None:
    series = default_df["Germany"].copy()

    splitter = DataSplitting(series)

    assert isinstance(splitter.data, pd.DataFrame)
    assert list(splitter.data.columns) == ["series"]
    expected = series.copy()
    expected.name = "series"
    pd.testing.assert_series_equal(splitter.data["series"], expected)


def test_slice_by_proportion(default_df: pd.DataFrame) -> None:
    series = default_df["Austria"]
    n = len(series)
    train_len = int(n * 0.6)
    val_len = int(n * 0.2)

    train, val, test = DataSplitting._slice_by_proportion(series, 0.6, 0.2)

    assert len(train) == train_len
    assert len(val) == val_len
    assert len(test) == n - train_len - val_len
    pd.testing.assert_series_equal(pd.concat([train, val, test]), series)
    if train_len:
        assert train.index[-1] == series.index[train_len - 1]
    if val_len:
        assert val.index[0] == series.index[train_len]
        assert val.index[-1] == series.index[train_len + val_len - 1]
    assert test.index[0] == series.index[train_len + val_len]


def test_split_by_proportion_method(default_df: pd.DataFrame) -> None:
    splitter = DataSplitting(default_df)
    n = len(default_df)
    train_len = int(n * 0.3)
    val_len = int(n * 0.2)

    result = splitter._split_by_proportion(0.3, 0.2)

    assert set(result.keys()) == set(default_df.columns)
    for col in default_df.columns:
        train, val, test = result[col]
        assert len(train) == train_len
        assert len(val) == val_len
        assert len(test) == n - train_len - val_len
        pd.testing.assert_series_equal(pd.concat([train, val, test]), default_df[col])


def test_split_by_proportion_public(default_df: pd.DataFrame) -> None:
    splitter = DataSplitting(default_df)
    n = len(default_df)
    train_len = int(n * 0.6)
    val_len = int(n * 0.2)

    result = splitter.split(0.6, 0.2)

    for col in default_df.columns:
        train, val, test = result[col]
        assert len(train) == train_len
        assert len(val) == val_len
        assert len(test) == n - train_len - val_len
        pd.testing.assert_series_equal(pd.concat([train, val, test]), default_df[col])


def test_split_by_proportion_with_default_val(default_df: pd.DataFrame) -> None:
    splitter = DataSplitting(default_df)
    n = len(default_df)
    train_len = int(n * 0.4)
    val_len = int(n * 0.4)

    result = splitter.split(0.4)

    for col in default_df.columns:
        train, val, test = result[col]
        assert len(train) == train_len
        assert len(val) == val_len
        assert len(test) == n - train_len - val_len
        pd.testing.assert_series_equal(pd.concat([train, val, test]), default_df[col])


def test_split_by_proportion_invalid_sum_raises(default_df: pd.DataFrame) -> None:
    splitter = DataSplitting(default_df)

    with pytest.raises(ValueError):
        splitter.split(0.6, 0.6)


def test_slice_by_date(default_df: pd.DataFrame) -> None:
    series = default_df["Switzerland"]
    train_date = default_df.index[24]
    val_date = default_df.index[72]
    idx_train = list(series.index).index(train_date)
    idx_val = list(series.index).index(val_date)

    train, val, test = DataSplitting._slice_by_date(series, train_date, val_date)

    assert len(train) == idx_train
    assert len(val) == idx_val - idx_train
    assert len(test) == len(series) - idx_val
    pd.testing.assert_series_equal(pd.concat([train, val, test]), series)
    assert train.index[-1] == series.index[idx_train - 1]
    assert val.index[0] == series.index[idx_train]
    assert test.index[0] == series.index[idx_val]


def test_split_by_date_method(default_df: pd.DataFrame) -> None:
    splitter = DataSplitting(default_df)
    train_date = default_df.index[12]
    val_date = default_df.index[48]
    idx_train = list(default_df.index).index(train_date)
    idx_val = list(default_df.index).index(val_date)

    result = splitter._split_by_date(train_date, val_date)

    for col in default_df.columns:
        train, val, test = result[col]
        assert len(train) == idx_train
        assert len(val) == idx_val - idx_train
        assert len(test) == len(default_df) - idx_val
        pd.testing.assert_series_equal(pd.concat([train, val, test]), default_df[col])


def test_split_by_date_public(default_df: pd.DataFrame) -> None:
    splitter = DataSplitting(default_df)
    train_date = default_df.index[6]
    val_date = default_df.index[30]
    idx_train = list(default_df.index).index(train_date)
    idx_val = list(default_df.index).index(val_date)

    result = splitter.split(train_date, val_date)

    for col in default_df.columns:
        train, val, test = result[col]
        assert len(train) == idx_train
        assert len(val) == idx_val - idx_train
        assert len(test) == len(default_df) - idx_val
        pd.testing.assert_series_equal(pd.concat([train, val, test]), default_df[col])


def test_split_by_date_with_default_val(default_df: pd.DataFrame) -> None:
    splitter = DataSplitting(default_df)
    train_date = default_df.index[18]
    idx_train = list(default_df.index).index(train_date)

    result = splitter.split(train_date)

    for col in default_df.columns:
        train, val, test = result[col]
        assert len(train) == idx_train
        assert len(val) == 0
        assert len(test) == len(default_df) - idx_train
        pd.testing.assert_series_equal(pd.concat([train, val, test]), default_df[col])


def test_split_mismatched_types_raises(default_df: pd.DataFrame) -> None:
    splitter = DataSplitting(default_df)

    with pytest.raises(ValueError):
        splitter.split(0.5, default_df.index[10])


def test_split_date_not_in_index_raises(default_df: pd.DataFrame) -> None:
    splitter = DataSplitting(default_df)

    with pytest.raises(ValueError):
        splitter.split(pd.Timestamp("1995-01-01"), pd.Timestamp("1996-01-01"))


def test_split_date_order_raises(default_df: pd.DataFrame) -> None:
    splitter = DataSplitting(default_df)
    train_date = default_df.index[20]
    val_date = default_df.index[10]

    with pytest.raises(ValueError):
        splitter.split(train_date, val_date)
