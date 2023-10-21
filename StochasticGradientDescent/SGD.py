import random
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
class SGD:
    def __init__(self, x, y, gradient, start=None, lr=0.05, seed=None, max_epochs=1000, delta=0.0001):
        """

        Args:
            x: array with the x-axis values
            y: array with the y-axis values
            gradient: function representing gradient
            start: starting point
            lr: learning rate
            seed: parameter used to make it less-random
            max_epochs: maximum number of iterations
            delta: value of step
        """
        self._x = x
        self._y = y
        self._gradient = gradient
        self._start = start
        self._lr = lr
        self._seed = seed if seed else random.randint(10, 101)
        self._max_epochs = max_epochs
        self._delta = delta

    def set_starting_points(self):
        """

        Returns:
            Randomly chosen starting points for the algorithm
        """
        random.seed(self._seed)
        random_index = random.randint(0, len(self._x) - 1)
        random_element_x = self._x[random_index]
        random_element_y = self._y[random_index]
        return np.array([random_element_x, random_element_y]).reshape((2, 1))

    def fit_point(self):
        """

        Returns:
                Fitted Point
        """
        current_epoch = 0
        curr_point = self.set_starting_points()
        while current_epoch < self._max_epochs:
            diff = self._lr * self._gradient(curr_point)
            if np.all(np.abs(diff) <= self._delta):
                break
            curr_point -= diff
            current_epoch += 1
        return curr_point

def gradient(v: np.array) -> np.array:
    """

    Args:
        v: (2,1) array with the x and y

    Returns:
        Array with the gradient parameters
    """
    dx = np.exp(-20 * v[0]**2 - 5 * v[0] +10 ) * v[1] * np.exp(-v[0] ** 2 - 0.5 * v[0] - v[1] ** 2)
    dy = -10 * v[0] * (2 * v[1] ** 2 - 1) * np.exp(-v[0] ** 2 - 0.5 * v[0] - v[1] ** 2)
    arr = np.array([dx, dy]).reshape((2, 1))
    return arr


def func(x: np.array, y: np.array) -> np.dtype:
    """

    Args:
        x: x-values array
        y: y-values array

    Returns:
        Array with the values of the function
    """
    return (10 * x * y) / np.exp(x ** 2 + 0.5 * x + y ** 2)


def display_3d_function(x:np.array,y:np.array, func: Callable, result_point = None) -> None:
    """

    Args:
        x: x-values array
        y: y-values array
        func: function to be displayed
        result_point: point calculated with the SGD algorithm

    Performs:
        Display function plot with the calculated point
    """
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    ax = plt.subplot(projection="3d", computed_zorder=False)
    ax.plot_surface(X, Y, Z, cmap="viridis", zorder=0)
    if result_point is not None:
        ax.scatter(result_point[0], result_point[1], func(result_point[0], result_point[1]), color="black", zorder=1)
    plt.show()

def main():
    x = np.arange(-5, 5, 0.01)
    y = np.arange(-5, 5, 0.01)
    lr = 0.05
    seed = 1
    sgd = SGD(x=x, y=y, gradient=gradient, start=None, lr=lr, seed=None, max_epochs=100000, delta=0.000001)
    result_point = sgd.fit_point()
    display_3d_function(x,y, func,result_point)


if __name__ == "__main__":
    main()
