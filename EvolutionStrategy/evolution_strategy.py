import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from typing import Callable

class EvolutionStrategy:
    def __init__(self, mi, lamb, starting_point=None):
        self._mi = mi
        self._lamb = lamb
        self._logger = logging.getLogger(__name__)
        self._starting_point = starting_point

    def set_starting_points(self):
        """
        Selects random starting points for the algorithm.

        Returns:
            Randomly chosen starting points for the algorithm - randomly chosen starting points
        """
        self._logger.info("Choose starting points")
        random.seed(self._seed)
        random_indexes = random.choices(range(0, len(self._x) - 1), k=2)
        x = self._x[random_indexes[0]]
        y = self._y[random_indexes[1]]
        self._logger.info(f"Starting points: ({x},{y})")
        return np.array([x, y]).reshape((2, 1))


def func(x: np.array, y: np.array) -> np.dtype:
    """
    Calculates the function's value based on the x and y arrays.

    Args:
        x: x-values array - array with x values
        y: y-values array - array with y values

    Returns:
        Array with the values of the function - array with function values
    """
    return (10 * x * y) / np.exp(x ** 2 + 0.5 * x + y ** 2)


def display_3d_function(x: np.array, y: np.array, function: Callable, result_point=None) -> None:
    """
    Displays a three-dimensional function plot.

    Args:
        x: x-values array - array with x values
        y: y-values array - array with y values
        function: function to be displayed - function to be displayed on the plot
        result_point: point calculated with the SGD algorithm - point calculated with the SGD algorithm (optional)

    Performs:
        Display function plot with the calculated point - Displays the function plot with the calculated point
    """
    x_arr, y_arr = np.meshgrid(x, y)
    z_arr = function(x_arr, y_arr)
    ax = plt.subplot(projection="3d", computed_zorder=False)
    ax.plot_surface(x_arr, y_arr, z_arr, cmap="cool", zorder=0)
    if result_point is not None:
        ax.scatter(result_point[0], result_point[1], func(result_point[0], result_point[1]), color="red", zorder=1)
    plt.show()


def set_logging(folder_name: str) -> None:
    """
    Set logger configuration

    Args:
        folder_name: folder with log files


    """
    os.makedirs(folder_name, exist_ok=True)
    logging.basicConfig(filename=f"{folder_name}/app_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.log",
                        level=logging.INFO)


def main():
    x = np.arange(-3, 3, 0.01)
    y = np.arange(-3, 3, 0.01)
    set_logging("report")
    evolution = EvolutionStrategy(mi =1, lamb =1)
    display_3d_function(x, y, func)


if __name__ == "__main__":
    main()
