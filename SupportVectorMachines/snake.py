from enum import Enum


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Snake:
    def __init__(self, block_size, bounds):
        self.length = None
        self.direction = None
        self.body = None
        self.head_color = (0, 150, 0)
        self.color = (0, 200, 0)
        self.block_size = block_size
        self.bounds = bounds
        self.respawn()

    def respawn(self):
        self.body = [(self.block_size, i*self.block_size) for i in range(3)]
        self.length = len(self.body)
        self.direction = Direction.DOWN

    def draw(self, game, window):
        for segment in self.body[:-1]:
            game.draw.rect(window, self.color, (segment[0], segment[1], self.block_size, self.block_size))
        head = self.body[-1]
        game.draw.rect(window, self.head_color, (head[0], head[1], self.block_size, self.block_size))

    def move(self):
        curr_head = self.body[-1]
        if self.direction == Direction.DOWN:
            next_head = (curr_head[0], curr_head[1] + self.block_size)
            self.body.append(next_head)
        elif self.direction == Direction.UP:
            next_head = (curr_head[0], curr_head[1] - self.block_size)
            self.body.append(next_head)
        elif self.direction == Direction.RIGHT:
            next_head = (curr_head[0] + self.block_size, curr_head[1])
            self.body.append(next_head)
        elif self.direction == Direction.LEFT:
            next_head = (curr_head[0] - self.block_size, curr_head[1])
            self.body.append(next_head)

        if self.length < len(self.body):
            self.body.pop(0)

    def turn(self, direction):
        if self.direction == Direction.DOWN and direction != Direction.UP:
            self.direction = direction
        elif self.direction == Direction.UP and direction != Direction.DOWN:
            self.direction = direction
        elif self.direction == Direction.LEFT and direction != Direction.RIGHT:
            self.direction = direction
        elif self.direction == Direction.RIGHT and direction != Direction.LEFT:
            self.direction = direction

    def eat(self):
        self.length += 1

    def check_for_food(self, food):
        head = self.body[-1]
        if head[0] == food.x and head[1] == food.y:
            self.eat()
            food.respawn()
            return 1.001  # This is for reinforcement learning
        return 0

    def is_tail_collision(self):
        head = self.body[-1]
        has_eaten_tail = False

        for i in range(len(self.body) - 1):
            segment = self.body[i]
            if head[0] == segment[0] and head[1] == segment[1]:
                has_eaten_tail = True

        return has_eaten_tail

    def is_wall_collision(self):
        head = self.body[-1]
        if 0 <= head[0] < self.bounds[0] and 0 <= head[1] < self.bounds[1]:
            return False
        return True
