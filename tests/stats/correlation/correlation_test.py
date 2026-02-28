import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr, kendalltau

from reidfo.stats.correlation.pearson import PearsonCorrelation
from reidfo.stats.correlation.spearman import SpearmanCorrelation
from reidfo.stats.correlation.kendall import KendallTauCorrelation


@pytest.fixture
def df():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 10.0],
        },
        index=index,
    )


def test_pearson_matrix_shape_and_index(df):
    matrix = PearsonCorrelation(df).compute_matrix()
    assert list(matrix.index) == ["a", "b", "c"]
    assert list(matrix.columns) == ["a", "b", "c"]


def test_pearson_matrix_values(df):
    matrix = PearsonCorrelation(df).compute_matrix()
    cols = ["a", "b", "c"]
    for c1 in cols:
        for c2 in cols:
            expected = pearsonr(df[c1], df[c2]).statistic
            assert np.isclose(matrix.at[c1, c2], expected)


def test_pearson_matrix_is_symmetric(df):
    matrix = PearsonCorrelation(df).compute_matrix()
    assert np.allclose(matrix.values, matrix.values.T, equal_nan=True)


def test_pearson_matrix_diagonal_is_one(df):
    matrix = PearsonCorrelation(df).compute_matrix()
    for col in ["a", "b", "c"]:
        assert np.isclose(matrix.at[col, col], 1.0)


def test_pearson_caches_result(df):
    obj = PearsonCorrelation(df)
    first = obj.compute_matrix()
    second = obj.compute_matrix()
    assert first is second


def test_pearson_scores_none_before_compute(df):
    obj = PearsonCorrelation(df)
    assert obj.scores is None


def test_pearson_subset_no_cache(df):
    obj = PearsonCorrelation(df)
    result = obj.compute_matrix(labels1=["a"], labels2=["b", "c"])
    assert result.shape == (1, 2)
    assert obj.scores is None


def test_pearson_validation_rejects_non_datetime_index(df):
    bad_df = df.reset_index(drop=True)
    with pytest.raises(ValueError):
        PearsonCorrelation(bad_df)


def test_pearson_validation_rejects_non_string_columns(df):
    bad_df = df.copy()
    bad_df.columns = [0, 1, 2]
    with pytest.raises(ValueError):
        PearsonCorrelation(bad_df)


def test_pearson_nan_handling(df):
    nan_df = df.copy()
    nan_df.loc[nan_df.index[0], "a"] = float("nan")
    matrix = PearsonCorrelation(nan_df).compute_matrix()
    assert np.isfinite(matrix.at["a", "b"])


def test_spearman_matrix_values(df):
    matrix = SpearmanCorrelation(df).compute_matrix()
    cols = ["a", "b", "c"]
    for c1 in cols:
        for c2 in cols:
            expected = spearmanr(df[c1], df[c2]).statistic
            assert np.isclose(matrix.at[c1, c2], expected)


def test_spearman_caches_result(df):
    obj = SpearmanCorrelation(df)
    first = obj.compute_matrix()
    second = obj.compute_matrix()
    assert first is second


def test_kendall_matrix_values(df):
    matrix = KendallTauCorrelation(df).compute_matrix()
    cols = ["a", "b", "c"]
    for c1 in cols:
        for c2 in cols:
            expected = kendalltau(df[c1], df[c2]).statistic
            assert np.isclose(matrix.at[c1, c2], expected)


def test_kendall_caches_result(df):
    obj = KendallTauCorrelation(df)
    first = obj.compute_matrix()
    second = obj.compute_matrix()
    assert first is second
