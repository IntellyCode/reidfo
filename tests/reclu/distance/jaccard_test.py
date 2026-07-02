import numpy as np
import pandas as pd

from reidfo.reclu.distance.jaccard import JaccardDistance


def _labels():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    return {
        "a": pd.Series([0, 0, 1, 1, 0, 0], index=idx),
        "b": pd.Series([0, 0, 1, 1, 0, 0], index=idx),
        "c": pd.Series([0, 1, 1, 0, 0, 1], index=idx),
    }


def test_distance_matrix_shape_and_symmetry():
    matrix = JaccardDistance(_labels()).get_distance_matrix(window=1)
    assert list(matrix.index) == ["a", "b", "c"]
    assert list(matrix.columns) == ["a", "b", "c"]
    np.testing.assert_allclose(matrix.values.astype(float), matrix.values.T.astype(float))


def test_distance_matrix_diagonal_is_zero():
    matrix = JaccardDistance(_labels()).get_distance_matrix(window=1)
    for key in matrix.index:
        assert matrix.loc[key, key] == 0


def test_identical_series_have_zero_distance():
    matrix = JaccardDistance(_labels()).get_distance_matrix(window=1)
    assert matrix.loc["a", "b"] == 0


def test_window_at_series_start_does_not_wrap_to_series_end():
    # Regime change at the first position with window > 0 previously wrapped via
    # negative iloc slicing and could match against unrelated changes at the tail.
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    labels = {
        "a": pd.Series([0, 1, 1, 1, 1, 1], index=idx),
        "b": pd.Series([1, 1, 1, 1, 1, 0], index=idx),
    }
    matrix = JaccardDistance(labels).get_distance_matrix(window=2)
    assert matrix.loc["a", "b"] == 1
