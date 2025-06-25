import pandas as pd
import numpy as np


# --------------------- Feature Computation Functions ---------------------
# reviewed
def calculate_downside(series: pd.Series, hl: float) -> pd.Series:
    """
    Computes the exponentially weighted moving downside deviation for the given series.

    Args:
        series (pd.Series, optional): The input rating change series.
        hl (float, optional): The halflife parameter for computing the downside deviation.
    """
    series_neg = np.minimum(series, 0.0)
    sq_mean = series_neg.pow(2).ewm(halflife=hl).mean()
    return np.sqrt(sq_mean)


# reviewed
def calculate_ewm_mean(series: pd.Series, hl: float) -> pd.Series:
    """
    Computes the exponentially weighted moving mean of the rating change series.

    Args:
        series (pd.Series, optional): The input rating change series.
        hl (float, optional): The halflife parameter for computing the EWM mean.
    """
    return series.ewm(halflife=hl).mean()


# reviewed
def calculate_log_downside_deviation(series: pd.Series, hl: float) -> pd.Series:
    """
    Computes the natural logarithm of the exponentially weighted moving downside deviation.

    Args:
        series (pd.Series, optional): The input rating change series.
        hl (float, optional): The halflife parameter for computing the downside deviation.
    """
    dd = calculate_downside(series, hl)
    dd_safe = dd.replace(0, 1e-20)
    return np.log(dd_safe)


# reviewed
def calculate_ewm_sortino_ratio(series: pd.Series, hl: float) -> pd.Series:
    """
    Computes the exponentially weighted moving Sortino ratio of the rating change series.

    Args:
        series (pd.Series, optional): The input rating change series.
        hl (float, optional): The halflife parameter for computing the EWM mean and downside deviation.
    """
    ewm_mean = calculate_ewm_mean(series, hl)
    dd = calculate_downside(series, hl)
    dd_safe = dd.replace(0, 1e-20)
    return ewm_mean.div(dd_safe)


# --------------------- v1 Feature Computation Functions ---------------------
# reviewed
def feature_observation(series: pd.Series) -> pd.Series:
    """
    Returns the original observation series.

    Args:
        series (pd.Series, optional): The input time series.

    Returns:
        pd.Series: The original observation series.
    """
    return series


# reviewed
def feature_absolute_change(series: pd.Series) -> pd.Series:
    """
    Computes the absolute change of the time series.

    Args:
        series (pd.Series, optional): The input time series.

    Returns:
        pd.Series: The absolute change series, computed as |x[t] - x[t-1]|.
    """
    return series.diff().abs()


# reviewed
def feature_previous_absolute_change(series: pd.Series) -> pd.Series:
    """
    Computes the previous absolute change of the time series.

    Args:
        series (pd.Series, optional): The input time series.

    Returns:
        pd.Series: The previous absolute change series, computed as |x[t-1] - x[t-2]|.
    """
    return series.diff().abs().shift(1)


# reviewed
def compute_centered_mean(series: pd.Series, w: int) -> pd.Series:
    """
    Computes the centered mean using a rolling window of length w.

    Args:
        series (pd.Series, optional): The input time series.
        w (int, optional): The window length.

    Returns:
        pd.Series: The centered mean series computed over [x[t-w+1], ..., x[t]].
    """
    return series.rolling(window=w).mean()


# reviewed
def compute_centered_std(series: pd.Series, w: int) -> pd.Series:
    """
    Computes the centered standard deviation using a rolling window of length w.

    Args:
        series (pd.Series, optional): The input time series.
        w (int, optional): The window length.

    Returns:
        pd.Series: The centered standard deviation series computed over [x[t-w+1], ..., x[t]].
    """
    return series.rolling(window=w).std()


# reviewed
def compute_left_mean(series: pd.Series, w: int) -> pd.Series:
    """
    Computes the left window mean, i.e. the mean over the window
    [x_{t-w+1}, ..., x_{t-w/2}].

    Args:
        series (pd.Series, optional): The input time series.
        w (int, optional): The full window length (must be even).

    Returns:
        pd.Series: The left window mean series.
    """
    L = w // 2
    # Rolling window of full length w, then take the first half of the window.
    return series.rolling(window=w, min_periods=w).apply(
        lambda x: np.mean(x[:L]), raw=True
    )


# reviewed
def compute_left_std(series: pd.Series, w: int) -> pd.Series:
    """
    Computes the left window standard deviation, i.e. the std over the window
    [x_{t-w+1}, ..., x_{t-w/2}].

    Args:
        series (pd.Series, optional): The input time series.
        w (int, optional): The full window length (must be even).

    Returns:
        pd.Series: The left window standard deviation series.
    """
    L = w // 2
    return series.rolling(window=w, min_periods=w).apply(
        lambda x: np.std(x[:L], ddof=1), raw=True
    )


# reviewed
def compute_right_mean(series: pd.Series, w: int) -> pd.Series:
    """
    Computes the right window mean, i.e. the mean over the window
    [x_{t-w/2+1}, ..., x_{t}].

    Args:
        series (pd.Series, optional): The input time series.
        w (int, optional): The full window length (must be even).

    Returns:
        pd.Series: The right window mean series.
    """
    L = w // 2
    return series.rolling(window=w, min_periods=w).apply(
        lambda x: np.mean(x[L:]), raw=True
    )


# reviewed
def compute_right_std(series: pd.Series, w: int) -> pd.Series:
    """
    Computes the right window standard deviation, i.e. the std over the window
    [x_{t-w/2+1}, ..., x_{t}].

    Args:
        series (pd.Series, optional): The input time series.
        w (int, optional): The full window length (must be even).

    Returns:
        pd.Series: The right window standard deviation series.
    """
    L = w // 2
    return series.rolling(window=w, min_periods=w).apply(
        lambda x: np.std(x[L:], ddof=1), raw=True
    )

def compute_slope(series: pd.Series, w: int) -> pd.Series:
    """
    Computes the slope of the best-fit line over each rolling window of length w.

    Assumes the x-values are evenly spaced (0, 1, 2, ...), which is typical for time series data sampled at regular intervals.

    Args:
        series (pd.Series): The input time series.
        w (int): The window length.

    Returns:
        pd.Series: A series of slopes computed for each window.
    """
    def slope(y_vals):
        if len(y_vals) < 2:
            return 0.0
        x_vals = np.arange(len(y_vals))
        s, _ = np.polyfit(x_vals, y_vals, 1)
        return s

    return series.rolling(window=w, min_periods=w).apply(slope, raw=True)


def compute_filtered_slope(series: pd.Series, w: int) -> pd.Series:
    """
    Computes the slope of the best-fit line over each rolling window of length w,
    filtering out outliers (points more than 3 standard deviations from the mean).

    Assumes the x-values are evenly spaced (0, 1, 2, ...), which is typical for time series data sampled at regular intervals.

    Args:
        series (pd.Series): The input time series.
        w (int): The window length.

    Returns:
        pd.Series: A series of slopes computed for each window.
    """
    def slope(y_vals):
        mean = np.mean(y_vals)
        std = np.std(y_vals, ddof=1)
        mask = np.abs(y_vals - mean) <= 3 * std
        filtered_y = y_vals[mask]
        if len(filtered_y) < 2:
            return 0.0
        x_vals = np.arange(len(filtered_y))
        s, _ = np.polyfit(x_vals, filtered_y, 1)
        return s

    return series.rolling(window=w, min_periods=w).apply(slope, raw=True)
