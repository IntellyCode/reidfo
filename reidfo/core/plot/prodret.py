import datetime as dt

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter

from .util import check_axes, filter_date_range

def plot_prodret(ret_series: pd.Series,
                 start_date: dt.datetime = None,
                 end_date: dt.datetime = None,
                 ax: Axes = None,
                 ylabel: str = "Cumulative Returns") -> Axes:
    """
    Plot cumulative returns for a return series.

    :param ret_series: The return series to plot.
    :param start_date: Optional start date for filtering.
    :param end_date: Optional end date for filtering.
    :param ax: Matplotlib axes to plot on.
    :param ylabel: Label for the y-axis.
    :return: The matplotlib axes with the plot.
    """
    if ret_series.iloc[0] != 0:
        ret_series = _prepend_zero_return(ret_series)
    ax = check_axes(ax)
    ret_series = _prepare_returns_series(ret_series, start_date, end_date)
    _plot_cumulative_returns(ret_series, ax)
    ax.set(ylabel=ylabel)
    _convert_yaxis_to_percent(ax)
    return ax


def _prepend_zero_return(ret_series: pd.Series) -> pd.Series:
    name = ret_series.name
    first_idx = ret_series.index[0]
    freq = pd.infer_freq(ret_series.index)
    offset = pd.tseries.frequencies.to_offset(freq)
    new_idx = first_idx - offset
    prepend = pd.Series([0.0], index=[new_idx])
    ret_series = pd.concat([prepend, ret_series]).sort_index()
    ret_series.name = name
    return ret_series


def _prepare_returns_series(ret_series: pd.Series,
                            start_date: dt.datetime,
                            end_date: dt.datetime) -> pd.Series:
    ret_series = filter_date_range(ret_series, start_date, end_date)
    ret_series.index.name = None
    return ret_series


def _plot_cumulative_returns(ret_series: pd.Series, ax: Axes) -> None:
    ((1 + ret_series).cumprod() - 1).plot(ax=ax)


def _convert_yaxis_to_percent(ax: Axes) -> None:
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
