import pandas as pd
from typing import List


def detect_label_changes(labels: pd.Series) -> List[pd.Timestamp]:
    """
    Identify the dates in a time series where the label changes.

    :param labels: A pandas Series indexed by dates, containing regime labels.
    :return: A list of dates where the label changes in the time series.
    """
    labels = labels.dropna()
    change_dates = labels[labels != labels.shift()].index[1:]
    return list(change_dates)