import os
import multiprocessing as mp
from typing import List, Dict, Any, Union, Callable

import numpy as np
import pandas as pd
from IPython.display import clear_output
from hyperopt import Trials

from .util import optimize_single_series
from .clear_util import clear_console
from src.reidfo.optimisation.objectives.base_objective import BaseObjective
from src.reidfo.logging import setup_logger

logger = setup_logger(__name__)


class HyperoptMultiprocessor:
    def __init__(self,
                 objective: BaseObjective,
                 time_series_data: pd.DataFrame,
                 search_space: Dict[str, Any],
                 max_evals: int = 100,
                 timeout: int = None,
                 seed: int = 42,
                 n_jobs: int = None,
                 print_batch_size: int = 10,
                 ):
        """
        :param objective: Initialized Objective instance to copy
        :param time_series_data: A dataframe containing time series data to be optimized
        :param search_space: Search space dictionary
        :param max_evals: Max evals per series
        :param timeout: Timeout per series (secs)
        :param seed: RNG seed override per worker
        :param n_jobs: Parallel workers (defaults to cpu_count-1)
        :param print_batch_size: Print batch size when printing the queue
        """
        self.objective = objective
        self.time_series_data = time_series_data
        self.series_list = list(time_series_data.index)
        self.search_space = search_space
        self.max_evals = max_evals
        self.timeout = timeout
        self.seed = seed
        self.n_jobs = n_jobs if n_jobs is not None else max(1, os.cpu_count() - 1)
        self.print_batch_size = print_batch_size

    def run(self, func: Callable = optimize_single_series) -> Dict[str, Union[Trials, str]]:
        """
        Run hyperopt on each series in parallel, showing a Rich Table
        where each row has its own ProgressBar.
        """
        logger.info(f"Running {func.__name__}")
        manager = mp.Manager()
        queue = manager.Queue()

        objectives = []
        for series in self.series_list:
            obj = self.objective.clone(series)
            obj.set_series(self.time_series_data.loc[series])
            objectives.append(obj)

        jobs_args = [
            (
                obj,
                self.search_space.copy(),
                self.max_evals,
                self.timeout,
                self.seed,
                False,
                queue,
            )
            for obj in objectives
        ]
        bars = self._make_table(self.series_list)
        final_results: Dict[str, Union[Trials, str]] = {}
        logger.info(f"Initializing {len(objectives)} objectives")
        with mp.Pool(processes=self.n_jobs) as pool:
            result_obj = pool.starmap_async(func, jobs_args)
            update_counter = 0
            recent_updates = set()

            while not result_obj.ready():
                while not queue.empty():
                    msg = queue.get_nowait()
                    series = msg["series"]
                    if "error" in msg:
                        error = msg["error"]
                        logger.error(f"Error in series '{series}': {type(error).__name__}: {str(error)}")
                    else:
                        completed = msg["completed"]
                        tps = msg["sec_per_trial"]
                        bars[series] = (completed, np.round(tps, 3))
                        update_counter += 1
                        recent_updates.add(series)

                if update_counter >= self.print_batch_size:
                    clear_output()
                    clear_console()
                    for series in sorted(list(recent_updates))[-self.n_jobs:]:
                        progress = f"{bars[series][0]}/{self.max_evals}"
                        logger.info(f"{series:<15} | {progress} | {bars[series][1]} s/trial")

                    update_counter = 0
                    recent_updates.clear()

            results = result_obj.get()
            for result in results:
                series = result["series"]
                trials_or_err = result.get("trials")
                if isinstance(trials_or_err, Trials):
                    final_results[series] = trials_or_err
                else:
                    final_results[series] = str(trials_or_err)
        return final_results

    @staticmethod
    def _make_table(series_list: List[str]) -> Dict[str, Any]:
        """
        Create a Bars Dictionary

        :param series_list: names of each series
        :returns bars dictionary
        """
        bars = {}
        for series in series_list:
            bars[series] = (0, 0)
        return bars
