import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib.ticker import PercentFormatter

from reidfo.core.plot.prodret import plot_prodret


def test_plot_prodret_basic():
    series = pd.Series([0.0, 0.1, -0.05], index=pd.date_range("2020-01-01", periods=3))
    ax = plot_prodret(series)
    ydata = ax.lines[0].get_ydata()
    expected = ((1 + series).cumprod() - 1).to_numpy()
    assert list(ydata) == list(expected)
    assert isinstance(ax.yaxis.get_major_formatter(), PercentFormatter)


def test_plot_prodret_date_filtering():
    dates = pd.date_range("2020-01-01", periods=3)
    series = pd.Series([0.02, 0.03, -0.01], index=dates)
    ax = plot_prodret(series, start_date=dates[1], end_date=dates[2])
    ydata = ax.lines[0].get_ydata()
    expected = ((1 + series.loc[dates[1]:dates[2]]).cumprod() - 1).to_numpy()
    assert list(ydata) == list(expected)
