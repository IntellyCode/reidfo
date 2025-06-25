from typing import Dict, List, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform



def _sort_dates(regime_changes: Dict[str, List[pd.Timestamp]],
                colours: Optional[List[str]] = None,
                monochrome: bool = False):
    all_dates = sorted({d for dates in regime_changes.values() for d in dates})
    time_series_names = list(regime_changes.keys())
    n_series = len(time_series_names)

    if monochrome:
        color = colours[0] if colours and len(colours) == 1 else 'red'
        colours = [color] * n_series
    elif colours is None:
        colours = [None] * n_series
    else:
        assert len(colours) == n_series, "Length of colours must match number of time series."
    return all_dates, time_series_names, n_series, colours

def plot_cluster_line(regime_changes: Dict[str, List[pd.Timestamp]],
                      colours: Optional[List[str]] = None,
                      monochrome: bool = False) -> None:
    """
    Plot regime change points over time for multiple time series on a single horizontal line.

    :param regime_changes: Dictionary with keys as time series names and values as lists of regime change dates.
    :param colours: Optional list of colours for each time series.
    :param monochrome: If True, use the same color for all series.
    """
    all_dates, time_series_names, n_series, colours = _sort_dates(regime_changes, colours, monochrome)

    fig, ax = plt.subplots()
    for i, (ts_name, ts_dates) in enumerate(regime_changes.items()):
        y = np.zeros(len(ts_dates))
        ax.plot(ts_dates, y, 'o', color=colours[i], ms=10)

    ax.set_yticks([0])
    ax.set_yticklabels(["All"])
    ax.set_ylim(-1, 1)
    ax.set_xlim(min(all_dates), max(all_dates))
    ax.set_xlabel("Date")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()


def plot_cluster_plane(regime_changes: Dict[str, List[pd.Timestamp]],
                       colours: Optional[List[str]] = None,
                       monochrome: bool = False) -> None:
    """
    Plot regime change points over time for multiple time series on separate horizontal lines.

    :param regime_changes: Dictionary with keys as time series names and values as lists of regime change dates.
    :param colours: Optional list of colours for each time series.
    :param monochrome: If True, use the same color for all series.
    """
    all_dates, time_series_names, n_series, colours = _sort_dates(regime_changes, colours, monochrome)

    fig, ax = plt.subplots(figsize=(24, 20))
    for i, (ts_name, ts_dates) in enumerate(regime_changes.items()):
        y = np.full(len(ts_dates), n_series - i - 1)
        ax.plot(ts_dates, y, 'o', color=colours[i], ms=10)

    ax.set_yticks(range(n_series))
    ax.set_yticklabels(sorted(time_series_names, reverse=True))
    ax.set_ylim(-1, n_series)
    ax.set_xlim(min(all_dates), max(all_dates))
    ax.set_xlabel("Date")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()


def plot_cluster_spikes(regime_changes: Dict[str, List[pd.Timestamp]]) -> None:
    """
    Plot a spike graph showing how many times a regime change occurs on each date.

    :param regime_changes: Dictionary with keys as time series names and values as lists of regime change dates.
    """
    all_dates = [d for dates in regime_changes.values() for d in dates]
    date_counts = pd.Series(all_dates).value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.plot(date_counts.index, date_counts.values, marker='o', linestyle='-')

    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Regime Changes")
    ax.set_yticks(range(0, date_counts.max() + 1))
    ax.grid(True, axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

def plot_dendrogram(distance_matrix: Union[pd.DataFrame, np.array], ylabel="Distance") -> None:
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    plt.figure()
    dendrogram(linkage_matrix, labels=distance_matrix.index.tolist(), leaf_rotation=90)
    plt.ylabel(ylabel)
    plt.xticks(fontsize=20)
    plt.tight_layout()