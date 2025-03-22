import random
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment


class Obstacle:

    def __init__(self, x, y, enviroment):
        self.position = (x, y)
        self.env = enviroment


    def move(self):
        possible_moves = [
            (self.position[0] - 1, self.position[1]),  # Up
            (self.position[0] + 1, self.position[1]),  # Down
            (self.position[0], self.position[1] - 1),  # Left
            (self.position[0], self.position[1] + 1),  # Right
            (self.position[0] - 1, self.position[1] - 1),  # Up-Left
            (self.position[0] - 1, self.position[1] + 1),  # Up-Right
            (self.position[0] + 1, self.position[1] - 1),  # Down-Left
            (self.position[0] + 1, self.position[1] + 1)  # Down-Right
        ]

        valid_moves = [(nx, ny) for nx, ny in possible_moves
                   if (0 <= nx < self.env.width and 0 <= ny < self.env.height)
                   and (nx, ny) not in [(agent.x, agent.y) for agent in self.env.agents]]
        move_prob = random.random()
        if move_prob < 0.5:
            if valid_moves != []:
                self.position = random.choice(valid_moves)
        

    
