import matplotlib
matplotlib.use("Agg")

import os
import matplotlib.pyplot as plt
import pytest

from reidfo.stats.copulas import ClaytonCopula


def test_fit_populates_copulas_for_all_pairs(price_df):
    cop = ClaytonCopula(price_df)
    cop.fit()
    assert len(cop._copulas) == len(cop._pairs)


def test_copulas_keys_match_pairs(price_df):
    cop = ClaytonCopula(price_df)
    cop.fit()
    assert set(cop._copulas.keys()) == set(cop._pairs)


def test_plot_single_pair_returns_figure(price_df):
    cop = ClaytonCopula(price_df)
    cop.fit()
    result = cop.plot(pairs=("A", "B"), show=False)
    assert isinstance(result, plt.Figure)
    plt.close("all")


def test_plot_multiple_pairs_returns_list(price_df):
    cop = ClaytonCopula(price_df)
    cop.fit()
    result = cop.plot(pairs=[("A", "B"), ("A", "C")], show=False)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(f, plt.Figure) for f in result)
    plt.close("all")


def test_plot_all_pairs_returns_list(price_df):
    cop = ClaytonCopula(price_df)
    cop.fit()
    result = cop.plot(show=False)
    assert isinstance(result, list)
    assert len(result) == len(cop._pairs)
    plt.close("all")


def test_plot_saves_files_with_correct_names(price_df, tmp_path):
    cop = ClaytonCopula(price_df)
    cop.fit()
    cop.plot(pairs=("A", "B"), save_path=str(tmp_path), show=False)
    expected = tmp_path / "ClaytonCopula_A_vs_B.png"
    assert expected.exists()
    plt.close("all")
