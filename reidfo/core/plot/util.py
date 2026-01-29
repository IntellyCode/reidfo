import datetime as dt

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


# reviewed
def matplotlib_setting():
    """
    Set global rcParams for matplotlib to produce nice and large publication-quality figures.
    """
    plt.rcParams['figure.figsize'] = (24, 12)
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['legend.fontsize'] = 30
    plt.rcParams['font.size'] = 26
    plt.rcParams['font.family'] = 'cmr10'
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    return


def check_axes(ax: Axes | None) -> Axes:
    """
    Ensures there is a matplotlib Axes to plot on.

    :param ax: Optional matplotlib Axes.
    :return: A matplotlib Axes instance.
    """
    if ax is None:
        _, ax = plt.subplots()
    return ax


def filter_date_range(
    series: pd.Series,
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
) -> pd.Series:
    """
    Filter a series by an optional start and end date.

    :param series: The series to filter.
    :param start_date: Optional start datetime.
    :param end_date: Optional end datetime.
    :return: Filtered series.
    """
    if start_date is None and end_date is None:
        return series
    return series.loc[start_date:end_date]
