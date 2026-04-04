import pandas as pd
import datetime as dt
from typing import Optional
from loguru import logger

from reidfo.core.preprocessing import clip_by_std, filter_date_range, standard_scale
from .collector.base_collector import BaseCollector
from .time_series_data import TimeSeriesData


# reviewed
class FeatureEngineer:
    def __init__(self,
                 df: pd.DataFrame,
                 clipper = clip_by_std,
                 scaler = standard_scale):
        """
        Extract a named time series from a time-indexed DataFrame, apply a collector,
        and return aligned TimeSeriesData.

        :param df: DataFrame with dates on the index and one column per named series.
        """
        self.df = df.copy()
        self.returns_df = self.df.pct_change(fill_method=None).iloc[1:]
        self.clipper = clipper
        self.scaler = scaler

    def _get_column(self, column: str, original: bool = False) -> pd.Series:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found.")
        return self.df[column] if original else self.returns_df[column]

    def get_data(self,
                 column: str,
                 collector: BaseCollector,
                 start_date: Optional[dt.date] = None,
                 end_date: Optional[dt.date] = None,
                 original: bool = False) -> TimeSeriesData:
        """
        Collect features using a provided collector instance.

        :param column: Column identifier of the requested series.
        :param collector: An instance of a BaseCollector subclass.
        :param start_date: Optional filtering start date.
        :param end_date: Optional filtering end date.
        :param original: Optional flag to indicate whether to use the original series or the return series.
        :return: TimeSeriesData object containing aligned series and features.
        """
        ts = self._get_column(column, original=original)
        featm = collector.collect(ts)
        featm = filter_date_range(featm, start_date, end_date)

        if self.clipper is not None:
            featm = self.clipper(featm)
        if self.scaler is not None:
            featm = self.scaler(featm)

        if featm.isnull().any().any():
            nan_cols = featm.columns[featm.isnull().any()].tolist()
            logger.warning(f"Feature matrix contains NaNs in columns: {nan_cols}")
            logger.debug(f"Preview of NaN columns:\n{featm[nan_cols].head()}")
            raise ValueError("Feature matrix contains NaNs.")

        series = filter_date_range(ts, start_date, end_date)
        return TimeSeriesData(series=series, feature_matrix=featm)
