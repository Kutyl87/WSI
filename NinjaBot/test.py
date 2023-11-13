import unittest

from main import run_game, GreedyAgent, MinMaxAgent


class TestMinMaxAgent(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMinMaxAgent, self).__init__(*args, **kwargs)

    def test_1(self):
        vector = [1, 5, 9000, 4, 3]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert len(first_agent.numbers) == 3
        assert len(second_agent.numbers) == 2
        assert sum(first_agent.numbers) > 9000

    def test_2(self):
        vector = [2, 0, 3, 1]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [2, 3]

    def test_3(self):
        vector = [0, 7, 0, 0]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [0, 7]

    def test_4(self):
        vector = [4, -9, 1, -8, -2, 8, -7, -4, 0, -1, 7, -5, 5, -3, -10, 3, 2, 6, -6, 9]
        first_agent, second_agent = MinMaxAgent(), MinMaxAgent()
        run_game(vector, first_agent, second_agent)
        assert sum(first_agent.numbers) == -1

    def test_5(self):
        vector = [0, 4, 6, 22, 1, 5, 8, 9, 3, 10, 7, 8]
        first_agent, second_agent = MinMaxAgent(), MinMaxAgent()
        run_game(vector, first_agent, second_agent)
        assert sum(first_agent.numbers) == 58

    def test_6(self):
        vector = [-42]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [-42]
        assert second_agent.numbers == []

    def test_7(self):
        vector = [6, 4, 8, 3, 1, 9, 6]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [6, 9, 8, 1]
        assert second_agent.numbers == [6, 4, 3]


if __name__ == '__main__':
    unittest.main()
