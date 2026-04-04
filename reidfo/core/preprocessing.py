import datetime as dt

import pandas as pd
from sklearn.preprocessing import StandardScaler


def filter_date_range(
    obj: pd.Series | pd.DataFrame,
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Filter a Series or DataFrame by an optional start and end date.

    :param obj: Series or DataFrame to filter.
    :param start_date: Optional start datetime.
    :param end_date: Optional end datetime.
    :return: Filtered Series or DataFrame.
    """
    if start_date is None and end_date is None:
        return obj
    return obj.loc[start_date:end_date]


def clip_by_std(
    df: pd.DataFrame,
    mul: float = 3.0,
) -> pd.DataFrame:
    """
    Clip each column to mean plus or minus a multiple of its standard deviation.

    :param df: Input feature matrix.
    :param mul: Standard deviation multiplier for clipping bounds.
    :return: Clipped feature matrix.
    """
    mean = df.mean(axis=0)
    std = df.std(axis=0, ddof=0)
    lower = mean - mul * std
    upper = mean + mul * std
    return df.clip(lower=lower, upper=upper, axis=1)


def standard_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each column and preserve pandas output.

    :param df: Input feature matrix.
    :return: Standardized feature matrix.
    """
    scaler = StandardScaler().set_output(transform="pandas")
    transformed = scaler.fit_transform(df)
    transformed.index = df.index
    return transformed
