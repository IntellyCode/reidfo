import numpy as np
import pandas as pd


# reviewed
def compute_downside_deviation(series: pd.Series, halflife: float) -> pd.Series:
    """
    Compute the exponentially weighted moving downside deviation.

    :param series: Input time series.
    :param halflife: Half-life parameter of the exponentially weighted window.
    :return: Downside deviation series.
    """
    series_neg = np.minimum(series, 0.0)
    sq_mean = series_neg.pow(2).ewm(halflife=halflife).mean()
    return np.sqrt(sq_mean)


# reviewed
def compute_ewm_mean(series: pd.Series, halflife: float) -> pd.Series:
    """
    Compute the exponentially weighted moving mean.

    :param series: Input time series.
    :param halflife: Half-life parameter of the exponentially weighted window.
    :return: Exponentially weighted moving mean series.
    """
    return series.ewm(halflife=halflife).mean()


# reviewed
def compute_log_downside_deviation(series: pd.Series, halflife: float) -> pd.Series:
    """
    Compute the logarithm of the downside deviation.

    :param series: Input time series.
    :param halflife: Half-life parameter of the exponentially weighted window.
    :return: Log downside deviation series.
    """
    dd = compute_downside_deviation(series, halflife)
    dd_safe = dd.replace(0, 1e-20)
    return np.log(dd_safe)


# reviewed
def compute_ewm_sortino_ratio(series: pd.Series, halflife: float) -> pd.Series:
    """
    Compute the exponentially weighted moving Sortino ratio.

    :param series: Input time series.
    :param halflife: Half-life parameter of the exponentially weighted window.
    :return: Exponentially weighted moving Sortino ratio series.
    """
    ewm_mean = compute_ewm_mean(series, halflife)
    dd = compute_downside_deviation(series, halflife)
    dd_safe = dd.replace(0, 1e-20)
    return ewm_mean.div(dd_safe)


# reviewed
def compute_observation(series: pd.Series) -> pd.Series:
    """
    Return the input series unchanged.

    :param series: Input time series.
    :return: Original observation series.
    """
    return series


# reviewed
def compute_absolute_change(series: pd.Series) -> pd.Series:
    """
    Compute the absolute one-step change.

    :param series: Input time series.
    :return: Absolute change series.
    """
    return series.diff().abs()


# reviewed
def compute_previous_absolute_change(series: pd.Series) -> pd.Series:
    """
    Compute the lagged absolute one-step change.

    :param series: Input time series.
    :return: Previous absolute change series.
    """
    return series.diff().abs().shift(1)


# reviewed
def compute_centered_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the rolling mean over the full window.

    :param series: Input time series.
    :param window: Rolling window length.
    :return: Rolling mean series.
    """
    return series.rolling(window=window).mean()


# reviewed
def compute_centered_std(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the rolling standard deviation over the full window.

    :param series: Input time series.
    :param window: Rolling window length.
    :return: Rolling standard deviation series.
    """
    return series.rolling(window=window).std()


# reviewed
def compute_left_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the mean over the left half of each rolling window.

    :param series: Input time series.
    :param window: Full rolling window length.
    :return: Left-half rolling mean series.
    """
    half_window = window // 2
    return series.rolling(window=window, min_periods=window).apply(
        lambda values: np.mean(values[:half_window]), raw=True
    )


# reviewed
def compute_left_std(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the standard deviation over the left half of each rolling window.

    :param series: Input time series.
    :param window: Full rolling window length.
    :return: Left-half rolling standard deviation series.
    """
    half_window = window // 2
    return series.rolling(window=window, min_periods=window).apply(
        lambda values: np.std(values[:half_window], ddof=1), raw=True
    )


# reviewed
def compute_right_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the mean over the right half of each rolling window.

    :param series: Input time series.
    :param window: Full rolling window length.
    :return: Right-half rolling mean series.
    """
    half_window = window // 2
    return series.rolling(window=window, min_periods=window).apply(
        lambda values: np.mean(values[half_window:]), raw=True
    )


# reviewed
def compute_right_std(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the standard deviation over the right half of each rolling window.

    :param series: Input time series.
    :param window: Full rolling window length.
    :return: Right-half rolling standard deviation series.
    """
    half_window = window // 2
    return series.rolling(window=window, min_periods=window).apply(
        lambda values: np.std(values[half_window:], ddof=1), raw=True
    )


def compute_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the slope of the best-fit line over each rolling window.

    :param series: Input time series.
    :param window: Rolling window length.
    :return: Rolling slope series.
    """
    def slope(values):
        if len(values) < 2:
            return 0.0
        x_values = np.arange(len(values))
        slope_value, _ = np.polyfit(x_values, values, 1)
        return slope_value

    return series.rolling(window=window, min_periods=window).apply(slope, raw=True)
