import pytest
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from reidfo.core.plot import plot_time_series, plot_prodret, plot_regimes


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


class TestPlotTimeSeries:
    def test_returns_axes(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(range(10), index=dates)
        ax = plot_time_series(series)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self):
        fig, ax = plt.subplots()
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(range(10), index=dates)
        result = plot_time_series(series, ax=ax)
        assert result is ax

    def test_sets_ylabel(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(range(10), index=dates)
        ax = plot_time_series(series, ylabel_data="Test Label")
        assert ax.get_ylabel() == "Test Label"

    def test_date_filtering_start(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(range(10), index=dates)
        ax = plot_time_series(series, start_date=dt.date(2020, 1, 5))
        lines = ax.get_lines()
        assert len(lines) > 0

    def test_date_filtering_end(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(range(10), index=dates)
        ax = plot_time_series(series, end_date=dt.date(2020, 1, 5))
        lines = ax.get_lines()
        assert len(lines) > 0

    def test_date_filtering_both(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.Series(range(10), index=dates)
        ax = plot_time_series(
            series,
            start_date=dt.date(2020, 1, 3),
            end_date=dt.date(2020, 1, 7)
        )
        lines = ax.get_lines()
        assert len(lines) > 0


class TestPlotProdret:
    def test_returns_axes(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, -0.01, 0.01], index=dates)
        ax = plot_prodret(returns)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self):
        fig, ax = plt.subplots()
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, -0.01, 0.01], index=dates)
        result = plot_prodret(returns, ax=ax)
        assert result is ax

    def test_sets_ylabel(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, -0.01, 0.01], index=dates)
        ax = plot_prodret(returns, ylabel_ret="Custom Label")
        assert ax.get_ylabel() == "Custom Label"

    def test_prepends_zero_when_first_value_nonzero(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=dates)
        ax = plot_prodret(returns)
        lines = ax.get_lines()
        assert len(lines) > 0

    def test_does_not_prepend_when_first_value_zero(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        returns = pd.Series([0.0, 0.02, -0.01, 0.03, 0.01], index=dates)
        ax = plot_prodret(returns)
        lines = ax.get_lines()
        assert len(lines) > 0

    def test_with_timestamp_index(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=dates)
        ax = plot_prodret(returns)
        assert isinstance(ax, plt.Axes)


class TestPlotRegimes:
    def test_returns_axes(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        regimes = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], index=dates)
        ax = plot_regimes(regimes)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self):
        fig, ax = plt.subplots()
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        regimes = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], index=dates)
        result = plot_regimes(regimes, ax=ax)
        assert result is ax

    def test_custom_colors(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        regimes = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], index=dates)
        ax = plot_regimes(regimes, colors_regimes=['blue', 'orange'])
        assert isinstance(ax, plt.Axes)

    def test_custom_labels(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        regimes = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], index=dates)
        ax = plot_regimes(regimes, labels_regimes=['Up', 'Down'])
        assert isinstance(ax, plt.Axes)

    def test_invalid_dimension_raises_valueerror(self):
        df = pd.DataFrame({
            "a": [0, 1, 0],
            "b": [1, 0, 1]
        })
        with pytest.raises(ValueError, match="expects a 1D label Series"):
            plot_regimes(df)

    def test_wrong_number_of_labels_raises_valueerror(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        regimes = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], index=dates)
        with pytest.raises(ValueError, match="Number of labels"):
            plot_regimes(regimes, labels_regimes=['Only One'])

    def test_insufficient_colors_raises_valueerror(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        regimes = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], index=dates)
        with pytest.raises(ValueError, match="Number of colors"):
            plot_regimes(regimes, colors_regimes=['g'])

    def test_none_colors_uses_color_cycle(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        regimes = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], index=dates)
        ax = plot_regimes(regimes, colors_regimes=None, labels_regimes=['Bull', 'Bear'])
        assert isinstance(ax, plt.Axes)

    def test_date_filtering(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        regimes = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], index=dates)
        ax = plot_regimes(
            regimes,
            start_date=dt.date(2020, 1, 3),
            end_date=dt.date(2020, 1, 8)
        )
        assert isinstance(ax, plt.Axes)

    def test_single_regime_block(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        regimes = pd.Series([0, 0, 0, 0, 0], index=dates)
        ax = plot_regimes(regimes, colors_regimes=['g'], labels_regimes=['Bull'])
        assert isinstance(ax, plt.Axes)

    def test_three_regimes(self):
        dates = pd.date_range("2020-01-01", periods=12, freq="D")
        regimes = pd.Series([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2], index=dates)
        ax = plot_regimes(
            regimes,
            colors_regimes=['g', 'r', 'b'],
            labels_regimes=['Bull', 'Bear', 'Neutral']
        )
        assert isinstance(ax, plt.Axes)
