import pandas as pd
import datetime as dt
from typing import Optional, Protocol

from sklearn.preprocessing import StandardScaler
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD
from jumpmodels.utils import filter_date_range

from src.reidfo.core.validation_utils import check_axis_is_date, check_axis_is_string
from .time_series_data import TimeSeriesData
from .collector.base_collector import BaseCollector
from src.reidfo.logging import setup_logger
logger = setup_logger(__name__)


class FitTransformable(Protocol):
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


# reviewed
class FeatureEngineer:
    def __init__(self, df: pd.DataFrame,
                 clipper: Optional[FitTransformable] = DataClipperStd(mul=3.0),
                 scaler: Optional[StandardScaler] = StandardScalerPD()):
        """
        Extracts rows from a date-indexed matrix, applies a user-defined collector to engineer features,
        and returns aligned TimeSeriesData. Rows must be string-indexed; columns must be datetime-like.

        :param df: DataFrame with rows as named entities and columns as dates.
        """
        check_axis_is_string(df, axis=0)
        check_axis_is_date(df, axis=1)
        self.df = df.copy()
        self.returns_df = self.df.pct_change(axis=1, fill_method=None).iloc[:, 1:]
        self.clipper = clipper
        self.scaler = scaler

    def _get_row(self, row: str, original: bool = False) -> pd.Series:
        if row not in self.df.index:
            raise ValueError(f"Row '{row}' not found.")
        return self.df.loc[row] if original else self.returns_df.loc[row]

    def get_data(self,
                 row: str,
                 collector: BaseCollector,
                 start_date: Optional[dt.date] = None,
                 end_date: Optional[dt.date] = None,
                 original: bool = False) -> TimeSeriesData:
        """
        Collect features using a provided collector instance.

        :param row: Row identifier (e.g., asset name).
        :param collector: An instance of a BaseCollector subclass.
        :param start_date: Optional filtering start date.
        :param end_date: Optional filtering end date.
        :param original: Optional flag to indicate whether to calculate features based on the orignal series or the return series
        :return: TimeSeriesData object containing aligned series and features.
        """
        ts = self._get_row(row, original=original)
        featm = collector.collect(ts)
        featm = filter_date_range(featm, start_date, end_date)

        if self.clipper is not None:
            featm = self.clipper.fit_transform(featm)
        if self.scaler is not None:
            featm = self.scaler.fit_transform(featm)

        if featm.isnull().any().any():
            nan_cols = featm.columns[featm.isnull().any()].tolist()
            logger.warning(f"Feature matrix contains NaNs in columns: {nan_cols}")
            logger.debug(f"Preview of NaN columns:\n{featm[nan_cols].head()}")
            raise ValueError("Feature matrix contains NaNs.")

        ret_ser = filter_date_range(ts, start_date, end_date)
        return TimeSeriesData(series=ret_ser, feature_matrix=featm)
