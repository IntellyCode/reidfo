from typing import List

import pandas as pd

from reidfo.core.validation_utils import check_index_is_datetime


def detect_label_changes(labels: pd.Series) -> List[pd.Timestamp]:
    """
    Identify the dates in a time series where the label changes.

    :param labels: A pandas Series indexed by dates, containing regime labels.
    :return: A list of dates where the label changes in the time series.
    """
    check_index_is_datetime(labels)
    labels = labels.dropna()
    change_dates = labels[labels != labels.shift()].index[1:]
    return list(change_dates)
