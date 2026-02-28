import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from reidfo.stats.copulas import GumbelCopula


def test_fit_populates_copulas_for_all_pairs(price_df):
    cop = GumbelCopula(price_df)
    cop.fit()
    # All fitted pairs must have a copula entry
    assert set(cop._copulas.keys()) == set(cop._pairs)


def test_copulas_keys_match_pairs(price_df):
    cop = GumbelCopula(price_df)
    cop.fit()
    assert set(cop._copulas.keys()) == set(cop._pairs)


def test_plot_single_pair_returns_figure(price_df):
    cop = GumbelCopula(price_df)
    cop.fit()
    if not cop._pairs:
        pytest.skip("All pairs had θ < 1; nothing to plot.")
    pair = cop._pairs[0]
    result = cop.plot(pairs=pair, show=False)
    assert isinstance(result, plt.Figure)
    plt.close("all")


def test_plot_all_pairs_returns_list(price_df):
    cop = GumbelCopula(price_df)
    cop.fit()
    if not cop._pairs:
        pytest.skip("All pairs had θ < 1; nothing to plot.")
    result = cop.plot(show=False)
    assert isinstance(result, list)
    assert len(result) == len(cop._pairs)
    plt.close("all")


def test_plot_saves_file(price_df, tmp_path):
    cop = GumbelCopula(price_df)
    cop.fit()
    if not cop._pairs:
        pytest.skip("All pairs had θ < 1; nothing to plot.")
    pair = cop._pairs[0]
    cop.plot(pairs=pair, save_path=str(tmp_path), show=False)
    expected = tmp_path / f"GumbelCopula_{pair[0]}_vs_{pair[1]}.png"
    assert expected.exists()
    plt.close("all")


def test_all_pairs_skipped_when_theta_lt_1():
    """When all Kendall τ produce θ < 1, _pairs is empty after fit and plot returns []."""
    # Negatively correlated series → τ < 0 → θ = 1/(1-τ) < 1
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    x = np.linspace(0, 1, 200)
    df = pd.DataFrame({"A": x, "B": -x + np.random.default_rng(1).normal(0, 0.001, 200)}, index=index)
    cop = GumbelCopula(df)
    cop.fit()
    assert cop._pairs == []
    result = cop.plot(show=False)
    assert result == []
