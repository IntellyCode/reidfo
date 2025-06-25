from hyperopt import hp
from typing import Callable
import itertools


# reviewed
def get_subsets(S: list, imp_keys=("obs",), min_size=3, max_size=12):
    """
    Computing all possible subsets of a set.
    :param S: The list of elements
    :param imp_keys: Elements that must be part of any subset
    :param min_size: Minimum subset size
    :param max_size: Maximum subset size
    :return: List of subsets
    """
    S_set = set(S)
    imp_set = set(imp_keys)
    if not imp_set.issubset(S_set):
        raise ValueError("All important keys must be present in S.")

    optional_keys = list(S_set - imp_set)
    min_optional = max(0, min_size - len(imp_set))
    max_optional = max_size - len(imp_set)
    results = []
    for r in range(min_optional, max_optional + 1):
        for combo in itertools.combinations(optional_keys, r):
            union_sorted = sorted(imp_set | set(combo))
            results.append(tuple(union_sorted))
    return results


def common_params(func_keys: list, add=""):
    return {
        'function_list': hp.choice(f'function_list_{add}', get_subsets(func_keys)),
        'halflives': hp.choice(f'halflives_{add}', [[3], [6], [12], [3, 6], [6, 12]]),
        'windows': hp.choice(f'windows_{add}', [[6], [12], [6, 12]])
    }


def regime_model_search_space(func_keys: list):
    options = [common_params(func_keys, add="rm")]
    return {
        "regime_model": {
            'feat_params': hp.choice('feat_params_rm', options)
        }
    }


def forecasting_model_search_space(func_keys: list):
    options = [common_params(func_keys, add="fm")]
    return {
        "forecasting_model": {
            'feat_params': hp.choice('feat_params_fm', options)
        }
    }


def hybrid_search_space(func_keys: list, rm_space: Callable, fm_space: Callable):
    params = {}
    params.update(rm_space(func_keys))
    params.update(fm_space(func_keys))
    return params


# reviewed
def lambda_search_space(add=""):
    return {
        "regime_model": {
            'lam': hp.quniform(f'lam_{add}', 0, 100, 0.001),
        }
    }


def halflife_search_space(add=""):
    return {
        "forecasting_model": {
            'smoothing_halflife': hp.quniform(f"smoothing_halflife_{add}", 0.001, 300, 0.001)
        }
    }


# reviewed
def statistical_jump_model_search_space(func_keys: list):
    params = {}
    params.update(regime_model_search_space(func_keys))
    params["regime_model"].update(lambda_search_space(add="sjm")["regime_model"])
    return params


def xgboost_search_space(func_keys: list):
    params = {}
    params.update(forecasting_model_search_space(func_keys))
    params["forecasting_model"].update(halflife_search_space(add="xgb")["forecasting_model"])
    return params
