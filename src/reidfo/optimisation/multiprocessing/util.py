import time
import numpy as np
import multiprocessing as mp
from typing import Dict, Any, Optional, Tuple, Type
from hyperopt import fmin, tpe, Trials


def send_progress_update(queue: mp.Queue, series: str, trials: Trials, start_time: time.time) -> Tuple[bool, Dict]:
    """
    Sends a progress update to the multiprocessing queue.

    :param queue: Multiprocessing queue for communication
    :param series: Series name being optimized
    :param trials: Hyperopt Trials object
    :param start_time: Start time
    :returns: Always false to not stop hyperopt optimisation
    """
    if queue is not None:
        elapsed = time.time() - start_time
        completed = len(trials.trials)
        sec_per_trial = elapsed / completed if completed > 0 else float('inf')
        queue.put({
            "series": series,
            "completed": len(trials.trials),
            "sec_per_trial": sec_per_trial,
        })
    return False, {}


def send_error(queue: mp.Queue, series: str, error: Exception):
    """
    Sends a progress update to the multiprocessing queue.

    :param queue: Multiprocessing queue for communication
    :param series: Series name being optimized
    :param error: Error message
    :returns: Always false to not stop hyperopt optimisation
    """
    if queue is not None:
        queue.put({
            "series": series,
            "error": error,
        })


def make_early_stop_fn(queue, series, start_time):
    if queue is None:
        return None
    return lambda trials_obj: send_progress_update(queue, series, trials_obj, start_time)


def optimize_single_series(objective: Any,
                           search_space: Dict[str, Any],
                           max_evals: int = 50,
                           timeout: Optional[int] = None,
                           seed: int = 42,
                           verbose: bool = False,
                           queue: Optional[Any] = None
                           ) -> Dict[str, Any]:
    """
    Runs hyperopt optimization for a single time series. Designed for multiprocessing.

    :param objective: Initialized Objective instance
    :param search_space: Hyperopt search space dictionary
    :param max_evals: Maximum number of evaluations
    :param timeout: Timeout in seconds
    :param seed: Random seed
    :param verbose: Whether to enable verbose output
    :param queue: Multiprocessing Queue to send progress updates (optional)
    :return: Dictionary with keys 'series' and 'trials'
    """
    np.random.seed(seed)
    try:
        start_time = time.time()
        trials = Trials()
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            timeout=timeout,
            trials=trials,
            rstate=np.random.default_rng(seed),
            verbose=verbose,
            early_stop_fn=make_early_stop_fn(queue, objective.series_name, start_time),
        )
        return {
            "series": objective.series_name,
            "trials": trials
        }

    except Exception as e:
        if queue is not None:
            send_error(queue, objective.series_name, e)
        else:
            print(f"Exception while optimizing {objective.series_name}: {e}")
        return {
            "series": objective.series_name,
            "trials": str(e)
        }

