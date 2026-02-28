import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def price_df():
    """Three positively correlated return series as a DatetimeIndex DataFrame.

    A shared common factor ensures positive Kendall τ between all pairs,
    which keeps Clayton θ > 0 and Gumbel θ > 1 for the test data.
    """
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    common = rng.normal(0, 0.01, 200).cumsum()
    data = {
        "A": common + rng.normal(0, 0.002, 200),
        "B": common + rng.normal(0, 0.002, 200),
        "C": common + rng.normal(0, 0.002, 200),
    }
    return pd.DataFrame(data, index=index)
