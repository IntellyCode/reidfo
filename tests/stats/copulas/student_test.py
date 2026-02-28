import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import pytest

from reidfo.stats.copulas import StudentTCopula


def test_fit_populates_copulas_for_all_pairs(price_df):
    cop = StudentTCopula(price_df)
    cop.fit()
    assert len(cop._copulas) == len(cop._pairs)


def test_copulas_keys_match_pairs(price_df):
    cop = StudentTCopula(price_df)
    cop.fit()
    assert set(cop._copulas.keys()) == set(cop._pairs)


def test_estimated_df_within_bounds(price_df):
    cop = StudentTCopula(price_df)
    cop.fit()
    for pair, cp in cop._copulas.items():
        assert 2 < cp.df <= 30, f"df={cp.df} out of (2, 30] for pair {pair}"


def test_correlation_matrix_is_symmetric_and_unit_diagonal(price_df):
    cop = StudentTCopula(price_df)
    cop.fit()
    for pair, cp in cop._copulas.items():
        corr = cp.corr
        assert corr.shape == (2, 2), f"corr shape {corr.shape} != (2,2) for {pair}"
        assert np.allclose(corr, corr.T), f"corr not symmetric for {pair}"
        assert np.allclose(np.diag(corr), 1.0), f"diagonal != 1 for {pair}"


def test_plot_single_pair_returns_figure(price_df):
    cop = StudentTCopula(price_df)
    cop.fit()
    result = cop.plot(pairs=("A", "B"), show=False)
    assert isinstance(result, plt.Figure)
    plt.close("all")


def test_plot_all_pairs_returns_list(price_df):
    cop = StudentTCopula(price_df)
    cop.fit()
    result = cop.plot(show=False)
    assert isinstance(result, list)
    assert len(result) == len(cop._pairs)
    plt.close("all")


def test_plot_saves_file(price_df, tmp_path):
    cop = StudentTCopula(price_df)
    cop.fit()
    cop.plot(pairs=("A", "B"), save_path=str(tmp_path), show=False)
    expected = tmp_path / "StudentTCopula_A_vs_B.png"
    assert expected.exists()
    plt.close("all")
