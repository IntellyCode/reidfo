import pandas as pd
import datetime as dt


# reviewed
class TimeSeriesData:
    def __init__(self, series: pd.Series, feature_matrix: pd.DataFrame):
        """
        :param series: The raw time series (e.g., returns).
        :param feature_matrix: Aligned feature matrix indexed by date.
        """
        self.series = series
        self.feature_matrix = feature_matrix

        if not self.series.index.equals(self.feature_matrix.index):
            raise ValueError("Series and feature matrix must have matching indices.")

    def get_series(self):
        return self.series

    def get_feature_matrix(self):
        return self.feature_matrix

    def get_index(self):
        return self.series.index

    def trim(self, start: dt.date|int, end: dt.date|int):
        """
        Trim the feature matrix and the time series
        :param start: Start date
        :param end: End date
        :return: None
        """
        if isinstance(start, dt.date) and isinstance(end, dt.date):
            self.series = self.series.loc[start:end]
            self.feature_matrix = self.feature_matrix.loc[start:end, :]
        elif isinstance(start, int) and isinstance(end, int):
            self.series = self.series.iloc[start:end]
            self.feature_matrix = self.feature_matrix.iloc[start:end, :]
        else:
            raise ValueError("start and end dates must have matching types.")
        return self
    def _check_compatibility(self, other: "TimeSeriesData") -> None:
        if not isinstance(other, TimeSeriesData):
            raise TypeError("Can add only TimeSeriesData objects\n"
                            f"Other type: {type(other)}")

        if type(self.series.index) is not type(other.series.index):
            raise TypeError("Index types differ.")

        if self.series.index[-1] >= other.series.index[0]:
            raise ValueError("Right-hand series must start after left-hand series ends.")

        if not self.feature_matrix.columns.equals(other.feature_matrix.columns):
            raise ValueError("Feature matrices must have identical columns in the same order.")

        if self.feature_matrix.index[-1] >= other.feature_matrix.index[0]:
            raise ValueError("Right-hand series must start after left-hand series ends.")

        if not (self.feature_matrix.dtypes.values == other.feature_matrix.dtypes.values).all():
            raise ValueError("Feature matrix column dtypes must match.")

        if self.series.dtype != other.series.dtype:
            raise ValueError("Series dtypes must match.")

    def __add__(self, other: "TimeSeriesData") -> "TimeSeriesData":
        """Return a new object with self first, then other."""
        self._check_compatibility(other)
        new_series = pd.concat([self.series, other.series])
        new_features = pd.concat([self.feature_matrix, other.feature_matrix])
        return TimeSeriesData(new_series, new_features)

    def __iadd__(self, other: "TimeSeriesData") -> "TimeSeriesData":
        """In-place extension with other, keeping self mutable."""
        self._check_compatibility(other)
        self.series = pd.concat([self.series, other.series])
        self.feature_matrix = pd.concat([self.feature_matrix, other.feature_matrix])
        return self

    def __repr__(self) -> str:
        return (f"Series: {self.series}\n"
                f"Features: {self.feature_matrix}")

    def __str__(self) -> str:
        return (f"Series: {self.series.shape}\n"
                f"Features: {self.feature_matrix.shape}")