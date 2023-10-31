import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from typing import Callable
from math import floor


class EvolutionStrategy:
    def __init__(self, x, y, funct, mi, lamb, max_iter, param, seed=None):
        """
        Initializes an EvolutionStrategy object with specified parameters.
        Args:
            x - list of x values
            y - list of y values
            funct - the objective function to be optimized
            mi - parent population size
            lamb - offspring population size
            max_iter - maximum number of iterations
            param - True if minimum is sought else False
            seed - random number generator seed (optional)
        """
        self._x = x
        self._y = y
        self._mi = mi
        self._lamb = lamb
        self._seed = seed if seed else random.randint(10, 101)
        self._logger = logging.getLogger(__name__)
        self._func = funct
        self._offset = 0.005
        self._max_iter = max_iter
        self._param = param

    def fitness(self, population):
        """
        Calculates the fitness of a population by applying the objective function to x and y values in the population.
        Args:
            population - a population where each individual is described as a pair (x, y)
        Returns:
            Standardized fitness values of the population
        """
        x = population[:, 0]
        y = population[:, 1]
        function_value = self._func(x, y)
        standardized_function_value = (function_value - np.min(function_value) + self._offset) / np.sum(
            function_value - np.min(function_value) + self._offset)
        return standardized_function_value

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
        return starting_points

    def extend_population(self, parents):
        """
        Extends the population by repeating parents.
        Args:
            parents - parent population
        Returns:
            Extended population
        """
        return np.repeat(parents, self._lamb, axis=0)

    def crossover(self, parents):
        """
        Performs crossover operation (not fully implemented).
        Args:
            parents - parent population
        Returns:
            Offspring population
        """
        a = np.repeat(np.random.normal(0, 1, size=(floor(self._mi / 2) * self._lamb, 1)), 2, axis=1)
        parents_extended = self.extend_population(parents)
        return parents_extended[:, 0, :] * a + parents_extended[:, 1, :] * (1 - a)

    @staticmethod
    def selection(population, probabilities):
        """
        Performs selection of individuals from the population based on their probabilities.
        Args:
            population - population to select from
            probabilities - probabilities of selection for each individual
        Returns:
            Selected individuals
        """
        return population[
            np.random.choice(population.shape[0], replace=False, p=probabilities,
                             size=(floor(len(population) / 2), 2))]

    def mutate(self, generation):
        """
        Applies mutation to a generation of individuals.
        Args:
            generation - population to mutate
        Returns:
            Mutated population
        """
        mut_prob = 0.1
        mutated_individuals = np.repeat(np.random.random(size=(generation.shape[0], 1)), 2, axis=1)
        gaussian_noise = np.random.normal(0, 1, size=(floor(self._mi / 2) * self._lamb, 2))
        return generation + (mutated_individuals < mut_prob) * gaussian_noise

    def get_new_population(self, whole_population, whole_population_rate):
        """
        Selects the top individuals from a combined population.
        Args:
            whole_population - combined population
            whole_population_rate - fitness values of the combined population
        Returns:
            New population and their normalized fitness values
        """
        sorted_indexes = np.argsort(whole_population_rate)[::self._param - (1-self._param)][:self._mi]
        new_population = whole_population[sorted_indexes]
        new_population_rate = whole_population_rate[sorted_indexes]
        return new_population, new_population_rate * 1 / sum(new_population_rate)

    def evolve(self):
        """
        Main evolution loop of the algorithm.
        Returns:
            The best individual found by the algorithm
        """
        i = 0
        population = self.set_starting_points(self._mi)
        population_rate = self.fitness(population)
        while i < self._max_iter:
            paired_population = self.selection(population=population, probabilities=population_rate)
            new_generation = self.crossover(parents=paired_population)
            mutated_new_generation = self.mutate(new_generation)
            whole_population = np.concatenate((population, mutated_new_generation), axis=0)
            whole_population_rate = self.fitness(whole_population)
            population, population_rate = self.get_new_population(whole_population, whole_population_rate)
            i += 1
        return population[0, :]


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
        result_point: point calculated with the Evolutionary algorithm - point calculated with the SGD algorithm (optional)

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
    x = np.arange(-7, 7, 0.01)
    y = np.arange(-7, 7, 0.01)
    set_logging("report")
    evolution = EvolutionStrategy(mi=1, lamb=1, x=x, y=y, seed=None, funct=func, max_iter=50, param=False)
    result_point = evolution.evolve()
    display_3d_function(x, y, func, result_point)


if __name__ == "__main__":
    main()
