import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import skew, kurtosis


class GeneralStatistics:
    def __init__(self, df: pd.DataFrame):
        """
        :param df: DataFrame with index as time and columns as series
        """
        self.df = df
        self.stats = None
        logger.success(
            "Initialized GeneralStatistics with %d rows and %d columns",
            df.shape[0],
            df.shape[1],
        )

    def compute(self) -> pd.DataFrame:
        """
        :return: DataFrame with descriptive statistics per column (series). Uses cached
            results if already computed.
        """
        if self.stats is not None:
            logger.info("Returning cached general statistics")
            return self.stats

        logger.info("Computing general statistics for %d columns", self.df.shape[1])
        stats_dict = {}

        for col in self.df.columns:
            ts = self.df[col].dropna().values
            if len(ts) < 2:
                stats_dict[col] = [np.nan] * 7
                continue

            mean = np.mean(ts)
            median = np.median(ts)
            std = np.std(ts, ddof=1)
            min_val = np.min(ts)
            max_val = np.max(ts)
            skewness = skew(ts)
            kurt = kurtosis(ts)

            stats_dict[col] = [mean, median, std, min_val, max_val, skewness, kurt]

        self.stats = pd.DataFrame.from_dict(
            stats_dict,
            orient="index",
            columns=["Mean", "Median", "StdDev", "Min", "Max", "Skew", "Kurt"],
        )
        logger.success("Computed general statistics for %d columns", len(self.stats))
        return self.stats
