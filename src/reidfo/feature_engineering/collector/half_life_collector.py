import pandas as pd

from ..util import calculate_ewm_mean, calculate_log_downside_deviation, calculate_ewm_sortino_ratio
from .base_collector import BaseCollector


# reviewed
class HalfLifeCollector(BaseCollector):
    def collect(self, time_series: pd.Series) -> pd.DataFrame:
        hls = self.feat_params.get("halflives", [5, 20, 60])
        if not isinstance(hls, (tuple, list)):
            raise TypeError("Params['halflives'] must be a tuple or list of integers")
        features = {}
        for hl in hls:
            features.update({
                f"ret_{hl}": calculate_ewm_mean(time_series, hl),
                f"DD-log_{hl}": calculate_log_downside_deviation(time_series, hl),
                f"sortino_{hl}": calculate_ewm_sortino_ratio(time_series, hl),
            })
        return pd.DataFrame(features)
