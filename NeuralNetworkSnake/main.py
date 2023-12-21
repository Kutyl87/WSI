import copy
import os
import pickle
import pygame
import time
import joblib
from food import Food
from model import game_state_to_data_sample, get_binaries
from snake import Snake, Direction
import numpy as np
from model import SVM, linear_kernel, rbf
import torch
from snakeNeuralNetwork import SnakeNeuralNetwork


def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)

    agent = BehavioralCloningAgent(block_size, bounds,
                                   "snake_neural_network.pt")  # Once your agent is good to go, change this line
    scores = []
    run = True
    pygame.time.delay(1000)
    while run and len(scores) <= 100:
        pygame.time.delay(1)  # Adjust game speed, decrease to test your agent and model quickly

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,  # The last element is snake's head
                      "snake_direction": snake.direction}

        direction = agent.act(game_state)
        snake.turn(direction)

        snake.move()
        snake.check_for_food(food)
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(300)
            scores.append(snake.length - 3)
            snake.respawn()
            food.respawn()

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    print(f"Scores: {scores}")
    print(f"Średnia wartość w 100 rozgrywkach: {sum(scores) / len(scores)}")
    agent.dump_data()
    pygame.quit()


class HumanAgent:
    """ In every timestep every agent should perform an action (return direction) based on the game state. Please note, that
    human agent should be the only one using the keyboard and dumping data. """

    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []

    def act(self, game_state) -> Direction:
        keys = pygame.key.get_pressed()
        action = game_state["snake_direction"]
        if keys[pygame.K_LEFT]:
            action = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            action = Direction.RIGHT
        elif keys[pygame.K_UP]:
            action = Direction.UP
        elif keys[pygame.K_DOWN]:
            action = Direction.DOWN

        self.data.append((copy.deepcopy(game_state), action))
        return action

    def dump_data(self):
        os.makedirs("data", exist_ok=True)
        current_time = time.strftime('%Y-%m-%d_%H:%M:%S')
        with open(f"data/{current_time}.pickle", 'wb') as f:
            pickle.dump({"block_size": self.block_size,
                         "bounds": self.bounds,
                         "data": self.data[:-10]},
                        f)  # Last 10 frames are when you press exit, so they are bad, skip them


class BehavioralCloningAgent:
    def __init__(self, block_size, bounds, model):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []
        self.model = torch.load(model)

    def act(self, game_state) -> Direction:
        print(game_state)
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        data_sample = get_binaries(game_state['snake_body'], self.bounds, self.block_size, game_state['food'])
        action = self.model(torch.tensor([data_sample]).to(torch.float32))
        argmax_dir = torch.argmax(action)
        return Direction(argmax_dir.item())
        # raise NotImplementedError()

    def dump_data(self):
        pass


if __name__ == "__main__":
    main()
