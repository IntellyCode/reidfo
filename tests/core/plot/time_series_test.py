import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from reidfo.core.plot.time_series import plot_time_series


def test_plot_time_series_creates_axes_and_sets_ylabel() -> None:
    series = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))

    ax = plot_time_series(series, ylabel_data="Test Label")

    assert ax.get_ylabel() == "Test Label"
    ydata = ax.lines[0].get_ydata()
    assert list(ydata) == [1, 2, 3]


def test_plot_time_series_date_filtering() -> None:
    dates = pd.date_range("2021-01-01", periods=5)
    series = pd.Series([10, 20, 30, 40, 50], index=dates)

    ax = plot_time_series(series, start_date=dates[1], end_date=dates[3])

    ydata = ax.lines[0].get_ydata()
    assert list(ydata) == [20, 30, 40]


def test_plot_time_series_uses_existing_axes() -> None:
    series = pd.Series([5, 6], index=pd.date_range("2022-01-01", periods=2))
    _, ax = plt.subplots()

    result = plot_time_series(series, ax=ax)

    assert result is ax
    assert len(ax.lines) == 1
