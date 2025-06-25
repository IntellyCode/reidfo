from .util import *

functional_dictionary = {
    "obs": feature_observation,
    "ab_ch": feature_absolute_change,
    "ab_pr_ch": feature_previous_absolute_change,
    "exp_do": calculate_downside,
    "ewm_me": calculate_ewm_mean,
    "log_exp_do": calculate_log_downside_deviation,
    #  "ewm_sor": calculate_ewm_sortino_ratio,
    "cen_me": compute_centered_mean,
    "cen_std": compute_centered_std,
    "le_me": compute_left_mean,
    "le_std": compute_left_std,
    "ri_me": compute_right_mean,
    "ri_std": compute_right_std,
    "slope": compute_slope,
    "mean_difference": lambda ts, w: compute_right_mean(ts, w) - compute_left_mean(ts,w),
}

keys = functional_dictionary.keys()
hls_keys = {"exp_do", "ewm_me", "log_exp_do", "ewm_sor"}
windows_keys = {"cen_me", "cen_std", "le_me", "le_std", "ri_me", "ri_std", "slope", "mean_difference"}
