import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from typing import Callable
from math import floor


class EvolutionStrategy:
    def __init__(self, x, y, func, mi, lamb, max_iter, seed=None):
        self._x = x
        self._y = y
        self._mi = mi
        self._lamb = lamb
        self._seed = seed if seed else random.randint(10, 101)
        self._logger = logging.getLogger(__name__)
        self._func = func
        self._offset = 0.005
        self._max_iter = max_iter
        self._paired_population = None
        self._new_generation = None

    def fitness(self):
        x = self._population[:, 0]
        y = self._population[:, 1]
        function_value = self._func(x, y)
        standarized_function_value = (function_value - np.min(function_value) + self._offset) / np.sum(
            function_value - np.min(function_value) + self._offset)
        return standarized_function_value

    def set_starting_points(self, mi):
        """
        Selects random starting points for the algorithm.

        Returns:
            Randomly chosen starting points for the algorithm - randomly chosen starting points
        """
        self._logger.info("Choose starting points")
        random.seed(self._seed)
        random_indexes = np.random.choice(range(0, len(self._x) - 1), size=(2, mi))
        x = self._x[random_indexes[0][:]].reshape(mi, 1)
        y = self._y[random_indexes[1][:]].reshape(mi, 1)
        starting_points = np.concatenate((x, y), axis=1)
        self._logger.info(f"Starting points: ({starting_points})")
        self._population = starting_points
        # return np.array([x, y]).reshape((2, 1))


    def extend_population(self):
        self._paired_population = np.repeat(self._paired_population, self._lamb, axis= 0)

    def crossover(self):
        a = np.repeat(np.random.normal(0, 1, size=(floor(self._mi/2) * self._lamb, 1)), 2, axis=1)
        # print(a.shape)
        self.extend_population()
        # print(self._population)
        # print(self._paired_population)
        print(self._paired_population.shape)
        self._new_generation = self._paired_population[:,0,:] * a + self._paired_population[:,1,:] * (1-a)
        print(self._new_generation.shape)
        # print(a.shape)

    def selection(self, probabilities):
        self._paired_population = self._population[
            np.random.choice(self._population.shape[0], replace=False, p=probabilities,
                             size=(floor(len(self._population) / 2), 2))]


    def mutate(self):
        mut_prob  = 0.1
        mutated_individuals = np.repeat(np.random.random(size = (self._new_generation.shape[0],1)),2,axis=1)
        gaussian_noise = np.random.normal(0, 1, size=(floor(self._mi/2) * self._lamb, 2))
        self._mutated_generation = self._new_generation + (mutated_individuals < mut_prob) * gaussian_noise
    def evolution(self):
        i = 0

        while i < self._max_iter:
            pass


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
    seed = 28
    set_logging("report")
    evolution = EvolutionStrategy(mi=120, lamb=3, x=x, y=y, seed=28, func=func, max_iter=1000)
    population = evolution.set_starting_points(120)
    fit = evolution.fitness()
    # evolution.mutate()
    evolution.selection(fit)
    evolution.crossover()
    evolution.mutate()
    # print(sel)
    # print(fit)
    # display_3d_function(x, y, func)


if __name__ == "__main__":
    main()
