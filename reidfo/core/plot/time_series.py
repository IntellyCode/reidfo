import datetime as dt
import pandas as pd
from matplotlib.axes import Axes

from .util import check_axes

def plot_time_series(
    data: pd.Series,
    start_date: dt.datetime = None,
    end_date: dt.datetime = None,
    ax: Axes | None = None,
    ylabel_data: str = "Variable",
) -> Axes:
    ax = check_axes(ax)
    series = data.loc[start_date:end_date]
    series.plot(ax=ax)
    ax.set_ylabel(ylabel_data)
    return ax
