import pandas as pd
import pytest

from reidfo.reclu.util import detect_label_changes


def test_detect_label_changes_returns_change_dates():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    labels = pd.Series([0, 0, 1, 1, 0], index=idx)
    changes = detect_label_changes(labels)
    assert changes == [idx[2], idx[4]]


def test_detect_label_changes_drops_nans_before_comparing():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    labels = pd.Series([0, None, 1, 1, 0], index=idx)
    changes = detect_label_changes(labels)
    assert changes == [idx[2], idx[4]]


def test_detect_label_changes_rejects_non_datetime_index():
    labels = pd.Series([0, 1, 0], index=["a", "b", "c"])
    with pytest.raises(ValueError):
        detect_label_changes(labels)
