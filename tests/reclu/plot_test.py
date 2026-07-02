import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from reidfo.reclu.plot import (
    plot_cluster_line,
    plot_cluster_plane,
    plot_cluster_spikes,
    plot_dendrogram,
)


def _regime_changes():
    dates_a = list(pd.date_range("2024-01-01", periods=3, freq="D"))
    dates_b = list(pd.date_range("2024-01-05", periods=2, freq="D"))
    return {"a": dates_a, "b": dates_b}


def test_plot_cluster_line_runs():
    plot_cluster_line(_regime_changes())


def test_plot_cluster_line_monochrome_runs():
    plot_cluster_line(_regime_changes(), colours=["blue"], monochrome=True)


def test_plot_cluster_plane_runs():
    plot_cluster_plane(_regime_changes())


def test_plot_cluster_spikes_runs():
    plot_cluster_spikes(_regime_changes())


def test_plot_cluster_line_rejects_mismatched_colours():
    with pytest.raises(AssertionError):
        plot_cluster_line(_regime_changes(), colours=["blue"], monochrome=False)


def test_plot_dendrogram_runs_with_dataframe():
    matrix = pd.DataFrame(
        [[0, 1, 2], [1, 0, 3], [2, 3, 0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
        dtype=float,
    )
    plot_dendrogram(matrix)


def test_plot_dendrogram_runs_with_ndarray():
    matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    plot_dendrogram(matrix)
