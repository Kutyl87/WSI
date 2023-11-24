import random
from math import inf
import time
from typing import List

random.seed(3)  # TODO: For final results set seed as your student's id modulo 42


class RandomAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if random.random() > 0.5:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class GreedyAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if vector[0] > vector[-1]:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class NinjaAgent:
    """   ⠀⠀⠀⠀⠀⣀⣀⣠⣤⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠴⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠠⠶⠶⠶⠶⢶⣶⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀
⠀⠀⠀⠀⢀⣴⣶⣶⣶⣶⣶⣶⣦⣬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
⠀⠀⠀⠀⣸⣿⡿⠟⠛⠛⠋⠉⠉⠉⠁⠀⠀⠀⠈⠉⠉⠉⠙⠛⠛⠿⣿⣿⡄⠀
⠀⠀⠀⠀⣿⠋⠀⠀⠀⠐⢶⣶⣶⠆⠀⠀⠀⠀⠀⢶⣶⣶⠖⠂⠀⠀⠈⢻⡇⠀
⠀⠀⠀⠀⢹⣦⡀⠀⠀⠀⠀⠉⢁⣠⣤⣶⣶⣶⣤⣄⣀⠀⠀⠀⠀⠀⣀⣾⠃⠀
⠀⠀⠀⠀⠘⣿⣿⣿⣶⣶⣶⣾⣿⣿⣿⡿⠿⠿⣿⣿⣿⣿⣷⣶⣾⣿⣿⡿⠀⠀
⠀⠀⢀⣴⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀
⠀⠀⣾⡿⢃⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀
⠀⢸⠏⠀⣿⡇⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀
⠀⠀⠀⢰⣿⠃⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⠛⠛⣉⣁⣤⡶⠁⠀⠀⠀⠀⠀
⠀⠀⣠⠟⠁⠀⠀⠀⠀⠀⠈⠛⠿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀
                かかって来い! """

    def __init__(OOOO000O000O00000):
        OOOO000O000O00000.numbers = []

    def act(O000000O000OO0O0O, O0OO0O0O0O0OO0O00: list):
        if len(O0OO0O0O0O0OO0O00) % 2 == 0:
            O00O0O0000000OO0O = sum(O0OO0O0O0O0OO0O00[::2])
            O0O00O0OO00O0O0O0 = sum(O0OO0O0O0O0OO0O00) - O00O0O0000000OO0O
            if O00O0O0000000OO0O >= O0O00O0OO00O0O0O0:
                O000000O000OO0O0O.numbers.append(O0OO0O0O0O0OO0O00[0])
                return O0OO0O0O0O0OO0O00[1:]  # explained: https://r.mtdv.me/articles/k1evNIASMp
            O000000O000OO0O0O.numbers.append(O0OO0O0O0O0OO0O00[-1])
            return O0OO0O0O0O0OO0O00[:-1]
        else:
            O00O0O0000000OO0O = max(sum(O0OO0O0O0O0OO0O00[1::2]), sum(O0OO0O0O0O0OO0O00[2::2]))
            O0O00O0OO00O0O0O0 = max(sum(O0OO0O0O0O0OO0O00[:-1:2]), sum(O0OO0O0O0O0OO0O00[:-2:2]))
            if O00O0O0000000OO0O >= O0O00O0OO00O0O0O0:
                O000000O000OO0O0O.numbers.append(O0OO0O0O0O0OO0O00[-1])
                return O0OO0O0O0O0OO0O00[:-1]
            O000000O000OO0O0O.numbers.append(O0OO0O0O0O0OO0O00[0])
            return O0OO0O0O0O0OO0O00[1:]


class MinMaxAgent:
    def __init__(self, max_depth=50):
        """
        Initialize the MinMaxAgent with the specified maximum depth for the Min-Max algorithm.

        Parameters:
        - max_depth (int): Maximum depth for the Min-Max algorithm.
        """
        self.numbers = []
        self.depth = max_depth

    @staticmethod
    def terminate(vector: List[int], depth: int) -> bool:
        """
        Check if the termination condition for the Min-Max recursion is met.

        Parameters:
        - vector (List[int]): Container with elements currently available in the game.
        - depth (int): Current depth of recursion.

        Returns:
        - bool: True if termination condition is met, False otherwise.
        """
        return len(vector) == 1 or depth <= 0

    @staticmethod
    def evaluate(state, vector, isMax):
        """
        Evaluate the current state of the game.

        Parameters:
        - state (List[List[int]]): Current state of the game.
        - vector (List[int]): Container with elements currently available in the game.
        - isMax (bool): Indicates whether the current player is maximizing.

        Returns:
        - int: Evaluation score for the current state.
        """
        return sum(state[0]) - sum(state[1]) - max(vector[0], vector[-1]) if isMax else sum(state[0]) - sum(
            state[1]) + max(vector[0], vector[-1])

    def minmax(self, state, vector, isMax, depth):
        """
         Implement the Min-Max algorithm.

         Parameters:
         - state (List[List[int]]): Current state of the game.
         - vector (List[int]): Container with elements currently available in the game.
         - is_max (bool): Indicates whether the current player is maximizing.
         - depth (int): Current depth of recursion.

         Returns:
         - int: The optimal evaluation score for the current state.
         """
        if self.terminate(vector, depth):
            return self.evaluate(state, vector, isMax)
        indexes = [0, len(vector) - 1]
        value = 0
        if isMax:
            maxEval = -inf
            for index in indexes:
                new_state = [state[0] + [vector[index]], state[1]]
                eval_result = self.minmax(new_state, vector[:index] + vector[index + 1:], False, depth - 1)
                maxEval = max(maxEval, eval_result)  # Update maxEval correctly
            return maxEval
        else:
            minEval = inf
            for index in indexes:
                value -= vector[index]
                new_state = [state[0], state[1] + [vector[index]]]
                eval_result = self.minmax(new_state, vector[:index] + vector[index + 1:], True, depth - 1)
                minEval = min(minEval, eval_result)  # Update minEval correctly
            return minEval

    def compare_start(self, first, last, vector):
        """
        Compare the results and return the updated vector.

        Parameters:
        - first (int): Evaluation score for starting with the first element.
        - last (int): Evaluation score for starting with the last element.
        - vector (List[int]): Container with elements currently available in the game.

        Returns:
        - List[int]: Updated vector based on the comparison of results.
        """
        if first > last:
            self.numbers.append(vector[0])
            return vector[1::]
        else:
            self.numbers.append(vector[-1])
            return vector[:-1]

    def act(self, vector: list):
        """
        Act based on the Min-Max algorithm.

        Parameters:
        - vector (List[int]): Container with elements currently available in the game.

        Returns:
        - List[int]: Updated vector after the agent's action.
        """
        state = [[], []]
        if len(vector) == 1:
            self.numbers.append(vector[-1])
            return vector[:-1]
        begin_first = vector[0] + self.minmax(state, vector[1:], False, self.depth - 1)
        begin_last = vector[-1] + self.minmax(state, vector[:-1], False, self.depth - 1)
        return self.compare_start(begin_first, begin_last, vector)


def run_game(vector, first_agent, second_agent):
    while len(vector) > 0:
        print(f"VC = {vector}")
        vector = first_agent.act(vector)
        if len(vector) > 0:
            vector = second_agent.act(vector)


def main():
    vector = [random.randint(-10, 10) for _ in range(16)]
    print(f"Vector: {vector}")
    first_agent, second_agent = NinjaAgent(), GreedyAgent()
    run_game(vector, first_agent, second_agent)

    print(f"First agent: {sum(first_agent.numbers)} Second agent: {sum(second_agent.numbers)}\n"
          f"First agent: {first_agent.numbers}\n"
          f"Second agent: {second_agent.numbers}")


if __name__ == "__main__":
    main()
