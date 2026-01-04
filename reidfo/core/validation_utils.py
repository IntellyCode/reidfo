import pandas as pd


# reviewed
def check_axis_is_date(obj: pd.DataFrame | pd.Series, axis: int = 0) -> None:
    """
    Validates that either the index (axis=0) or columns (axis=1) of a DataFrame
    are date-like and convertible to datetime.date.

    :param obj: The DataFrame or Series to check.
    :param axis: Axis to check — 0 for index, 1 for columns (series only accepts index)
    :raises ValueError: If conversion to datetime.date fails.
    """
    target = obj.index if axis == 0 else obj.columns
    try:
        _ = pd.to_datetime(target).date
    except Exception:
        name = "index" if axis == 0 else "columns"
        raise ValueError(f"{name.capitalize()} must be convertible to datetime.date.")


# reviewed
def check_axis_is_string(obj: pd.DataFrame | pd.Series, axis: int = 0) -> None:
    """
    Validates that either the index (axis=0) or columns (axis=1) of a DataFrame
    contain only strings.

    :param obj: The DataFrame or Series to check.
    :param axis: Axis to check — 0 for index, 1 for columns (series only accepts index)
    :raises ValueError: If the axis does not contain strings.
    """
    target = obj.index if axis == 0 else obj.columns
    if target.inferred_type != "string":
        raise ValueError("All column names must be strings.")


# reviewed
def check_df_for_nans(obj: pd.DataFrame | pd.Series) -> None:
    """
    Validates that the dataframe does not contain NaNs
    :param obj: The DataFrame or Series to check.
    :raises ValueError: If dataframe contains NaNs
    """
    if obj.isnull().values.any():
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
        n_regimes = labels.nunique(dropna=True)
        if n_regimes != required:
            raise ValueError(f"Exactly {required} regimes required; found {n_regimes}.")
    elif isinstance(labels, pd.DataFrame):
        regimes_per_row = labels.apply(lambda row: row.nunique(dropna=True), axis=1)
        if not regimes_per_row.eq(required).all():
            raise ValueError(f"Each row must contain exactly {required} regimes.")
    else:
        raise TypeError("Labels must be a pandas Series or DataFrame.")
