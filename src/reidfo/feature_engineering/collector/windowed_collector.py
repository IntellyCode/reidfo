import pandas as pd

from ..util import (
    feature_observation, feature_absolute_change, feature_previous_absolute_change,
    compute_centered_mean, compute_centered_std,
    compute_left_mean, compute_left_std,
    compute_right_mean, compute_right_std
)
from .base_collector import BaseCollector


# reviewed
class WindowedCollector(BaseCollector):
    def collect(self, time_series: pd.Series) -> pd.DataFrame:
        windows = self.feat_params.get("windows", [6, 12])
        if not isinstance(windows, (tuple, list)):
            raise TypeError("Params['windows'] must be a tuple or list of integers")

        features = {
            "observation": feature_observation(time_series),
            "abs_change": feature_absolute_change(time_series),
            "prev_abs_change": feature_previous_absolute_change(time_series),
        }
        for w in windows:
            features.update({
                f"centered_mean_{w}": compute_centered_mean(time_series, w),
                f"centered_std_{w}": compute_centered_std(time_series, w),
                f"left_mean_{w}": compute_left_mean(time_series, w),
                f"left_std_{w}": compute_left_std(time_series, w),
                f"right_mean_{w}": compute_right_mean(time_series, w),
                f"right_std_{w}": compute_right_std(time_series, w),
            })

        return pd.DataFrame(features)
