import pandas as pd


def convert_index_to_datetime(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Converts the index of a DataFrame or Series to datetime.

    :param df: The DataFrame or Series to convert.
    :return: A new DataFrame or Series with the converted datetime index.
    """
    result: pd.DataFrame | pd.Series = df.copy(deep=True)
    result.index = pd.to_datetime(result.index)
    return result
