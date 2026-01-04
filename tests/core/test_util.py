import pytest
import matplotlib.pyplot as plt

from reidfo.core.util import matplotlib_setting


class TestMatplotlibSetting:
    def test_sets_figure_size(self):
        matplotlib_setting()
        assert plt.rcParams['figure.figsize'] == [24, 12]

    def test_sets_title_size(self):
        matplotlib_setting()
        assert plt.rcParams['axes.titlesize'] == 30

    def test_sets_label_size(self):
        matplotlib_setting()
        assert plt.rcParams['axes.labelsize'] == 30

    def test_sets_tick_label_sizes(self):
        matplotlib_setting()
        assert plt.rcParams['xtick.labelsize'] == 30
        assert plt.rcParams['ytick.labelsize'] == 30

    def test_sets_legend_fontsize(self):
        matplotlib_setting()
        assert plt.rcParams['legend.fontsize'] == 30

    def test_sets_font_size(self):
        matplotlib_setting()
        assert plt.rcParams['font.size'] == 26

    def test_sets_font_family(self):
        matplotlib_setting()
        assert plt.rcParams['font.family'] == ['cmr10']

    def test_sets_mathtext(self):
        matplotlib_setting()
        assert plt.rcParams['axes.formatter.use_mathtext'] is True

    def test_sets_usetex(self):
        matplotlib_setting()
        assert plt.rcParams['text.usetex'] is True

    def test_sets_latex_preamble(self):
        matplotlib_setting()
        assert plt.rcParams['text.latex.preamble'] == r'\usepackage{amsmath}'

    def test_sets_savefig_dpi(self):
        matplotlib_setting()
        assert plt.rcParams['savefig.dpi'] == 300

    def test_sets_savefig_bbox(self):
        matplotlib_setting()
        assert plt.rcParams['savefig.bbox'] == 'tight'

    def test_returns_none(self):
        result = matplotlib_setting()
        assert result is None

    def test_idempotent(self):
        matplotlib_setting()
        matplotlib_setting()
        assert plt.rcParams['figure.figsize'] == [24, 12]