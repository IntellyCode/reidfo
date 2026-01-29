import datetime as dt
from typing import Dict, Hashable, List, Optional, Union
import pandas as pd

from .validation_utils import check_df_for_nans


class DataSplitting:
    def __init__(self, data: pd.Series | pd.DataFrame) -> None:
        """
        Initialize with a time series or dataframe.
        Series are converted to a single-column dataframe named "series".
        """
        self.data: pd.DataFrame = self._to_frame(data)
        check_df_for_nans(self.data)
    
    @staticmethod
    def _to_frame(data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            return data.to_frame(name="series")
        return data.copy()

    def split(self, train: float | dt.datetime, val: Optional[float | dt.datetime] = None) -> Dict[str, List[pd.Series]]:
        """
        Splits the data into train/test/validation sets.

        :param train: Either float (proportion) or date (index label)
        :param val: Optional. Same type as `train`. If None, treated as zero-length or full tail.
        :return: Dict mapping each column to [train, val, test].
        """
        if val is not None and type(train) != type(val):
            raise ValueError("train and val must be of the same type.")
        if isinstance(train, float):
            return self._split_by_proportion(train, val)
        return self._split_by_date(train, val)

    def _split_by_proportion(self, train: float, val: Optional[float]) -> Dict[str, List[pd.Series]]:
        val_float: float = train if val is None else val
        if train + val_float > 1:
            raise ValueError("Train and val proportions exceed 1.0")
        result: Dict[str, List[pd.Series]] = {
            col: self._slice_by_proportion(self.data[col], train, val_float)
            for col in self.data.columns
        }
        return result

    @staticmethod
    def _slice_by_proportion(series: pd.Series, train: float, val: float) -> List[pd.Series]:
        n: int = len(series)
        train_idx: int = int(n * train)
        val_idx: int = train_idx + int(n * val)
        return [series.iloc[:train_idx], series.iloc[train_idx:val_idx], series.iloc[val_idx:]]

    def _split_by_date(self, train: dt.datetime, val: dt.datetime) -> Dict[str, List[pd.Series]]:
        if train not in self.data.index or (val is not None and val not in self.data.index):
            raise ValueError("train/val must exist in the DataFrame index.")
        if val is None:
            val = train
        index_labels: List[Hashable] = list(self.data.index)
        if index_labels.index(val) < index_labels.index(train):
            raise ValueError("Expected date order: train < val (if val is provided).")
        result: Dict[Hashable, List[pd.Series]] = {
            col: self._slice_by_date(self.data[col], train, val)
            for col in self.data.columns
        }
        return result
    
    @staticmethod
    def _slice_by_date(series: pd.Series, train: dt.datetime, val: dt.datetime) -> List[pd.Series]:
        columns: List[Hashable] = list(series.index)
        idx_train: int = columns.index(train)
        idx_val: int = columns.index(val)
        return [series.iloc[:idx_train], series.iloc[idx_train:idx_val], series.iloc[idx_val:]]
