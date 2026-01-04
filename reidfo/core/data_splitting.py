from typing import Optional, Union
import datetime as dt
import pandas as pd

from .validation_utils import check_df_for_nans
from ..logging import setup_logger
logger = setup_logger(__name__)


# reviewed
class DataSplitting:
    def __init__(self, data: Union[pd.Series, pd.DataFrame]):
        """
        Initialize a DataSplitting object.

        :param data: Time series or dataframe to be split. All rows must have the same length. No NaNs allowed.
        :raises ValueError: If data contains NaNs or uneven lengths.
        """
        self.data = data.copy()
        check_df_for_nans(self.data)

    def split(self,
              train: Union[float, dt.date],
              val: Optional[Union[float, dt.date]] = None
              ) -> Union[list[pd.Series], dict[str, pd.DataFrame]]:
        """
        Splits the data into train/test/validation sets.

        :param train: Either float (proportion) or date (column/index label)
        :param val: Optional. Same type as `train`. If None, treated as zero-length or full tail.
        :return: List of Series [train, test, val] or dict of rows of the DataFrame.
        :raises ValueError: On mismatched types or out-of-order dates.
        """
        value_type = type(train)
        types = {value_type} | ({type(val)} if val is not None else set())

        if len(types) > 1:
            raise TypeError("All of train, test, and val (if provided) must be of the same type.")

        if value_type == float:
            logger.info("Splitting Dataframe by proportions")
            return self._split_by_proportion(train, val)
        elif value_type == dt.date:
            logger.info("Splitting Dataframe by dates")
            columns = self.data.columns if isinstance(self.data, pd.DataFrame) else self.data.index
            val_for_ordering = val if val is not None else columns[-1]
            if not (train < val_for_ordering):
                raise ValueError("Expected date order: train < val (if val is provided).")
            if train not in columns and val_for_ordering not in columns:
                raise ValueError("Expected dates to be in the columns\n"
                                 f"Columns: {columns}\n"
                                 f"Train: {train}\n"
                                 f"Val: {val_for_ordering}\n")
            return self._split_by_date(train, val)
        else:
            raise TypeError("train/test/val must all be floats or datetime.date or column labels.")

    @staticmethod
    def _slice_by_proportion(series, train: float, val: Optional[float]):
        n = len(series)
        train_idx = int(n * train)
        val_idx = train_idx + int(n * val)
        return [
            series.iloc[:train_idx],
            series.iloc[train_idx:val_idx],
            series.iloc[val_idx:]
        ]

    @staticmethod
    def _slice_by_date(series, train: dt.date, val: Optional[dt.date]):
        col_list = list(series.index)
        idx_train = col_list.index(train)
        idx_val = col_list.index(val) if val is not None else idx_train

        train_data = series.iloc[:idx_train]
        val_data = series.iloc[idx_train:idx_val]
        test_data = series.iloc[idx_val:]
        return [train_data, val_data, test_data]

    def _split_by_proportion(self, train: float, val: Optional[float]) -> Union[list, dict]:
        if val is None:
            val = 0.0

        total = train + val
        if total > 1:
            raise ValueError("Train and val proportions exceed 1.0")

        if isinstance(self.data, pd.Series):
            return self._slice_by_proportion(self.data, train, val)
        else:
            return {k: self._slice_by_proportion(self.data.loc[k, :], train, val) for k in self.data.index}

    def _split_by_date(self,
                       train: Union[dt.date, str],
                       val: Optional[Union[dt.date, str]]) -> Union[list, dict]:
        if isinstance(self.data, pd.Series):
            if train not in self.data.index or (val is not None and val not in self.data.index):
                raise ValueError("train/val must exist in the Series index.")
            return self._slice_by_date(self.data, train, val)
        else:
            if train not in self.data.columns or (val is not None and val not in self.data.columns):
                raise ValueError("train/val must exist in the DataFrame columns.")
            result = {}
            for row in self.data.index:
                row_series = self.data.loc[row]
                result[row] = self._slice_by_date(row_series, train, val)
            return result
