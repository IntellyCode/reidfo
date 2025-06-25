from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
from jumpmodels.plot import check_axes, convert_yaxis_to_percent
from jumpmodels.utils import filter_date_range
import datetime as dt


# reviewed
def plot_time_series(data: pd.Series,
                     start_date=None,
                     end_date=None,
                     ax=None,
                     ylabel_data="Variable",
                     ) -> plt.Axes:
    ax = check_axes(ax)
    data.loc[start_date:end_date].plot(ax=ax)
    ax.set_ylabel(ylabel_data)
    return ax


# reviewed
def plot_prodret(ret_series: pd.Series,
                 start_date=None,
                 end_date=None,
                 ax=None,
                 ylabel_ret="Cumulative Returns"
                 ) -> plt.Axes:
    if ret_series.iloc[0] != 0:
        name = ret_series.name
        first_idx = ret_series.index[0]
        freq = pd.infer_freq(ret_series.index)
        offset = pd.tseries.frequencies.to_offset(freq)

        if isinstance(first_idx, dt.date):
            new_idx = (pd.Timestamp(first_idx) - offset).date()
        else:
            new_idx = first_idx - offset
        prepend = pd.Series([0.0], index=[new_idx])
        ret_series = pd.concat([prepend, ret_series]).sort_index()
        ret_series.name = name
    ax = check_axes(ax)
    ret_df = filter_date_range(pd.DataFrame(ret_series), start_date, end_date)
    ret_df.index.name = None
    ((1 + ret_df).cumprod(axis=0) - 1).plot(ax=ax)

    ax.set(ylabel=ylabel_ret)
    convert_yaxis_to_percent(ax)
    return ax


def plot_regimes(regimes: pd.Series,
                 start_date: dt.date = None,
                 end_date: dt.date = None,
                 ax=None,
                 colors_regimes: Optional[list] = ['g', 'r'],
                 labels_regimes: Optional[list] = ['Bull', 'Bear']) -> plt.Axes:
    """
    Plot shaded regions for hard regime labels over full y-axis height.
    """
    regimes = filter_date_range(regimes, start_date, end_date)
    assert regimes.ndim == 1, "This version of plot_regimes expects a 1D label Series."

    n_c = len(set(regimes.dropna()))
    ax = check_axes(ax)

    if colors_regimes is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors_regimes = [color_cycle[i % len(color_cycle)] for i in range(n_c)]

    if labels_regimes is not None:
        assert len(labels_regimes) == n_c, "The number of labels must match the number of regimes."

    assert len(colors_regimes) >= n_c, "The number of colours passed must be equal to number of regimes"

    block_ids = (regimes != regimes.shift()).cumsum()
    blocks = [group for _, group in regimes.groupby(block_ids)]

    start = blocks[0].index[0]

    for block in blocks:
        end = block.index[-1]
        regime = block.iloc[0]
        color = colors_regimes[regime]
        label = labels_regimes[regime] if labels_regimes is not None else None
        ax.axvspan(start, end, color=color, alpha=0.3, label=label if label else None)

        start = end
    return ax
