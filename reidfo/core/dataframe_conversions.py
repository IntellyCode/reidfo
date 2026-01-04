import pandas as pd


# reviewed
def convert_axis_to_date(df: pd.DataFrame | pd.Series, axis: int = 0) -> pd.DataFrame | pd.Series:
    """
    Converts the index (axis=0) or columns (axis=1) of a DataFrame to datetime.date.

    :param df: The DataFrame or Series to convert.
    :param axis: Axis to convert — 0 for index, 1 for columns (series only accepts index)
    :return: A new DataFrame or Series with the converted index or columns.
    """
    result: pd.DataFrame | pd.Series = df.copy(deep=True)
    if axis == 0:
        result.index = pd.to_datetime(result.index).to_series().dt.date
    elif axis == 1:
        result.columns = pd.to_datetime(result.columns).to_series().dt.date
    else:
        raise ValueError("Axis must be 0 (index) or 1 (columns).")
    return result
