from typing import Dict, Any

import pandas as pd

from .base_collector import BaseCollector
from ..functional_dictionary import functional_dictionary, hls_keys, windows_keys


# reviewed
class CustomCollector(BaseCollector):
    def __init__(self, params: Dict[str, Any]):
        """
        :param params: Parameters for the feature engineering function.
        """
        super().__init__(params)
        if "feat_params" not in self.feat_params:
            raise ValueError("'feat_params' must be provided")

        self.feat_params = self.feat_params["feat_params"]

    def collect(self, time_series: pd.Series) -> pd.DataFrame:
        prefix = time_series.name if time_series.name is not None else "main"
        feat_dict = self._extract_features(time_series, self.feat_params, prefix)
        return pd.DataFrame(feat_dict)

    @staticmethod
    def _extract_features(series: pd.Series, feat_params: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        func_keys = list(feat_params.get("function_list", []))
        if not func_keys:
            raise ValueError("'function_list' must be provided in feat_params.")

        use_hls = set(hls_keys) & set(func_keys)
        use_windows = set(windows_keys) & set(func_keys)
        hls = feat_params.get("halflives", [5, 20, 60]) if use_hls else []
        windows = feat_params.get("windows", [6, 12]) if use_windows else []

        features = {}
        for key in func_keys:
            if key in hls_keys:
                for hl in hls:
                    features[f"{prefix}_{key}_{hl}"] = functional_dictionary[key](series, hl)
            elif key in windows_keys:
                for w in windows:
                    features[f"{prefix}_{key}_{w}"] = functional_dictionary[key](series, w)
            else:
                features[f"{prefix}_{key}"] = functional_dictionary[key](series)
        return features
