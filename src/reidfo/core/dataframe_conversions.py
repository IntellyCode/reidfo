import pandas as pd


# reviewed
def convert_axis_to_date(df: pd.DataFrame | pd.Series, axis: int = 0) -> pd.DataFrame:
    """
    Converts the index (axis=0) or columns (axis=1) of a DataFrame to datetime.date.

    :param df: The DataFrame or Series to convert.
    :param axis: Axis to convert — 0 for index, 1 for columns (series only accepts index)
    :return: A new DataFrame with the converted index or columns.
    """
    if axis == 0:
        df.index = pd.to_datetime(df.index).to_series().dt.date
    elif axis == 1:
        df.columns = pd.to_datetime(df.columns).to_series().dt.date
    else:
        raise ValueError("Axis must be 0 (index) or 1 (columns).")
    return df
