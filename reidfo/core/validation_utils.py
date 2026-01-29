from datetime import datetime

import pandas as pd


# reviewed
def check_index_is_datetime(obj: pd.DataFrame | pd.Series):
    """
    Validates that the index contains only datetime objects.

    :param obj: The DataFrame or Series to check.
    :raises ValueError: If the index does not contain only datetime objects.
    """
    index = obj.index
    if pd.api.types.is_datetime64_any_dtype(index):
        return
    if not all(isinstance(value, datetime) for value in index):
        raise ValueError("Index must contain only datetime objects.")



# reviewed
def check_columns_are_strings(obj: pd.DataFrame):
    """
    Validates that the DataFrame columns contain only strings.

    :param obj: The DataFrame to check.
    :raises ValueError: If columns are not strings.
    """
    if not isinstance(obj, pd.DataFrame):
        raise ValueError("Input must be a DataFrame.")
    if not all(isinstance(value, str) for value in obj.columns):
        raise ValueError("All columns must be strings.")


# reviewed
def check_df_for_nans(obj: pd.DataFrame | pd.Series):
    """
    Validates that the dataframe or series does not contain NaNs
    :param obj: The DataFrame or Series to check.
    :raises ValueError: If data contains NaNs
    """
    if isinstance(obj, pd.Series):
        has_nans = obj.isnull().any()
    else:
        has_nans = obj.isnull().to_numpy().any()
    if has_nans:
        raise ValueError("Input data contains NaNs.")


# reviewed
def validate_minimum_regimes(labels: pd.Series | pd.DataFrame, required: int = 2) -> None:
    """
    Validates that the label Series or DataFrame contains exactly `required` distinct regime values.

    :param labels: A Series or DataFrame of regime labels.
    :param required: The exact number of unique regimes required.
    :raises ValueError: If the number of regimes is not exactly equal to `required`.
    """
    if isinstance(labels, pd.Series):
        labels = labels.to_frame()
    regimes_per_column = labels.nunique(axis=0, dropna=True)
    if not regimes_per_column.eq(required).all():
        raise ValueError(f"Each column must contain exactly {required} regimes.")
