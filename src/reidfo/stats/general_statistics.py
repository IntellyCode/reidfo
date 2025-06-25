import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


class GeneralStatistics:
    def __init__(self, df: pd.DataFrame):
        """
        :param df: DataFrame with rows as time series and columns as timestamps
        """
        self.df = df
        self.stats = None

    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame with descriptive statistics per time series
        """
        stats_dict = {}

        for idx in self.df.index:
            ts = self.df.loc[idx].dropna().values
            if len(ts) < 2:
                stats_dict[idx] = [np.nan] * 7
                continue

            mean = np.mean(ts)
            median = np.median(ts)
            std = np.std(ts, ddof=1)
            min_val = np.min(ts)
            max_val = np.max(ts)
            skewness = skew(ts)
            kurt = kurtosis(ts)

            stats_dict[idx] = [mean, median, std, min_val, max_val, skewness, kurt]

        self.stats = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['Mean', 'Median', 'StdDev', 'Min', 'Max', 'Skew', 'Kurt'])
        return self.stats
