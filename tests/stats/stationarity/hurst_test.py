import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from reidfo.stats.stationarity.hurst import Hurst


def test_hurst_compute_columnwise_and_naming():
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "a": np.arange(1.0, 101.0),
            "b": np.where(np.arange(100) % 10 == 0, np.nan, np.arange(101.0, 201.0)),
            "c": [np.nan] * 99 + [1.0],
        },
        index=index,
    )

    hurst = Hurst(df)
    scores = hurst.compute()

    assert list(scores.index) == ["a", "b", "c"]
    assert list(scores.columns) == ["Hurst_exponent"]

    assert np.isfinite(scores.loc["a", "Hurst_exponent"])
    assert np.isfinite(scores.loc["b", "Hurst_exponent"])
    assert np.isnan(scores.loc["c", "Hurst_exponent"])

    df["a"] = np.arange(1000.0, 1100.0)
    cached = hurst.compute()

    assert cached is scores


def test_hurst_plot_saves_files(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "a": np.arange(1.0, 21.0),
            "b": np.arange(21.0, 41.0),
        },
        index=pd.date_range("2020-01-01", periods=20, freq="D"),
    )

    hurst = Hurst(df)
    hurst.compute()
    monkeypatch.setattr(Hurst, "compute", lambda self: (_ for _ in ()).throw(AssertionError()))

    hurst.plot(str(tmp_path))

    assert (tmp_path / "Hurst_a.pdf").exists()
    assert (tmp_path / "Hurst_b.pdf").exists()
