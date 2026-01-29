import datetime as dt
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

from .util import check_axes, filter_date_range


def plot_regimes(
    regimes: pd.Series,
    start_date: dt.datetime = None,
    end_date: dt.datetime = None,
    ax: Axes = None,
    regime_colors: Optional[Sequence] = ("g", "r"),
    regime_labels: Optional[Sequence] = ("Bull", "Bear"),
) -> plt.Axes:
    """
    Plot shaded regions for hard regime labels over the full y-axis height.

    :param regimes: Series of integer regime labels indexed by datetime.
    :param start_date: Optional start datetime for filtering the series.
    :param end_date: Optional end datetime for filtering the series.
    :param ax: Optional matplotlib Axes to draw on.
    :param regime_colors: Optional list of colors per regime id.
    :param regime_labels: Optional list of labels per regime id.
    :return: The matplotlib Axes with the regime shading.
    """
    regimes = filter_date_range(regimes, start_date, end_date)
    _validate_regimes(regimes)

    n_c = len(set(regimes.dropna()))
    ax = check_axes(ax)
    regime_colors = _resolve_colors(n_c, regime_colors)
    _validate_colors(regime_colors, n_c)
    _validate_labels(regime_labels, n_c)

    blocks = _build_blocks(regimes)
    _plot_blocks(ax, blocks, regime_colors, regime_labels)
    return ax


def _validate_regimes(regimes: pd.Series) -> None:
    if regimes.ndim != 1:
        raise ValueError("plot_regimes expects a 1D label Series.")


def _resolve_colors(n_c: int, regime_colors: Optional[Sequence]) -> list:
    if regime_colors is None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return [color_cycle[i % len(color_cycle)] for i in range(n_c)]
    return regime_colors


def _validate_colors(regime_colors: list, n_c: int) -> None:
    if len(regime_colors) < n_c:
        raise ValueError(
            f"Number of colors ({len(regime_colors)}) must be at least the number of regimes ({n_c})."
        )


def _validate_labels(regime_labels: Optional[Sequence], n_c: int) -> None:
    if regime_labels is not None and len(regime_labels) < n_c:
        raise ValueError(
            f"Number of labels ({len(regime_labels)})must be at least the number of regimes ({n_c})."
        )


def _build_blocks(regimes: pd.Series) -> list[pd.Series]:
    # Group consecutive regimes into blocks to shade continuous spans.
    block_ids = (regimes != regimes.shift()).cumsum()
    return [group for _, group in regimes.groupby(block_ids)]


def _plot_blocks(
    ax: plt.Axes,
    blocks: list[pd.Series],
    regime_colors: list,
    regime_labels: Optional[Sequence],
) -> None:

    start = blocks[0].index[0]
    for block in blocks:
        end = block.index[-1]
        regime = block.iloc[0]
        color = regime_colors[regime]
        label = regime_labels[regime] if regime_labels is not None else None
        ax.axvspan(start, end, color=color, alpha=0.3, label=label if label else None)
        start = end
