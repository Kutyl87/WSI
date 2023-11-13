import random
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
import logging
import datetime
import os


class SGD:
    def __init__(self, x: np.array, y: np.array, gradient_function: Callable, lr: float = 0.05, seed=None,
                 max_epochs=1000,
                 delta=0.0001):
        """
        Initialization of the SGD class, which implements the Stochastic Gradient Descent (SGD) algorithm.

        Args:
            x: array with the x-axis values - data on the X-axis
            y: array with the y-axis values - data on the Y-axis
            gradient_function: function representing the gradient - function representing the gradient
            lr: learning rate - learning rate
            seed: parameter used to make it less-random - random seed (optional)
            max_epochs: maximum number of iterations - maximum number of iterations
            delta: value of step - step size (optional)

        """
        self._logger = logging.getLogger(__name__)
        local_vars = locals()
        self._logger.info("Start the program. Log arguments")
        for var_name, var_value in local_vars.items():
            if var_name != 'self':
                self._logger.info(f"{var_name}: {var_value}")

        self._x = x
        self._y = y
        self._gradient = gradient_function
        self._lr = lr
        self._seed = seed if seed else random.randint(10, 101)
        self._max_epochs = max_epochs
        self._delta = delta

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

    def fit_point(self):
        """
        Conducts the point fitting procedure using the SGD algorithm.

        Returns:
            Fitted Point - fitted point
        """
        current_epoch = 0
        curr_point = self.set_starting_points()
        while current_epoch < self._max_epochs:
            diff = self._lr * self._gradient(curr_point)
            if np.all(np.abs(diff) <= self._delta):
                break
            curr_point += diff
            current_epoch += 1
        self._logger.info(f"Number of iterations: {current_epoch}")
        self._logger.info(f"Calculated points : {curr_point[0]},{curr_point[1]}")
        return curr_point


def gradient(v: np.array) -> np.array:
    """
    Calculates the gradient based on the (2,1) point v.

    Args:
        v: (2,1) array with the x and y - array with x and y values

    Returns:
        Array with the gradient parameters - array with gradient parameters
    """
    dx = (-20 * v[0] ** 2 - 5 * v[0] + 10) * v[1] * np.exp(-v[0] ** 2 - 0.5 * v[0] - v[1] ** 2)
    dy = -10 * v[0] * (2 * v[1] ** 2 - 1) * np.exp(-v[0] ** 2 - 0.5 * v[0] - v[1] ** 2)
    arr = np.array([dx, dy]).reshape((2, 1))
    return arr


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
    lr = 0.05
    max_epochs = 10000
    delta = 0.001
    set_logging("report")
    sgd = SGD(x=x, y=y, gradient_function=gradient, lr=lr, seed=28, max_epochs=max_epochs, delta=delta)
    result_point = sgd.fit_point()
    display_3d_function(x, y, func, result_point)


if __name__ == "__main__":
    main()
