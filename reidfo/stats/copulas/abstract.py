from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List
import pandas as pd
import matplotlib.pyplot as plt


class AbstractCopula(ABC):
    """
    Abstract base for all copula estimators.
    """

    @abstractmethod
    def __init__(self, data: pd.DataFrame, seed: int) -> None:
        """
        Initialize the copula with input data.

        :param data: Input time-series data. Index must be a DatetimeIndex;
                     columns must be string series names.
        """

    @abstractmethod
    def fit(self) -> None:
        """
        Fit copula parameters to the data.
        """

    @abstractmethod
    def plot(self,
             pairs: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None,
             n_samples: int = 10000,
             save_path: Optional[str] = None,
             show: bool = True) -> plt.Figure:
        """
        Plot the fitted copula.

        :param pairs: A tuple or list of tuples of (col1, col2) to plot.
                  If None or empty, plot all possible pairs.
        :param n_samples: Number of simulated samples to draw for the scatter plot.
        :param save_path: If provided, path to save the figure (e.g., 'clayton.png').
        :param show: Whether to display the plot immediately.
        :returns: The matplotlib Figure object or list of Figures.
        """
