import random


class Food:
    def __init__(self, block_size, bounds, lifetime=1e8):
        self.color = (150, 0, 0)
        self.x, self.y = 0, 0
        self.block_size = block_size
        self.bounds = bounds
        self.lifetime = lifetime
        self.time_left = lifetime
        self.respawn()

    def draw(self, game, window):
        game.draw.rect(window, self.color, (self.x, self.y, self.block_size, self.block_size))

    def respawn(self):
        self.time_left = self.lifetime
        blocks_in_x = (self.bounds[0]) / self.block_size
        blocks_in_y = (self.bounds[1]) / self.block_size
        self.x = random.randint(0, blocks_in_x - 1) * self.block_size
        self.y = random.randint(0, blocks_in_y - 1) * self.block_size

    def update(self):
        if random.random() < 0.01:
            self.color = tuple([random.randint(30, 220) for _ in range(3)])
        self.time_left -= 1
        if self.time_left < 0:
            self.time_left = self.lifetime
            self.respawn()
