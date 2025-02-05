import random
import numpy as np


class Agent:
    def __init__(self, id, x, y, enviroment):
        self.x = x
        self.y = y
        self.enviroment = enviroment
        self.id = id

        self.visited_cells = set()
        self.current_voronoi_cell = None

        self.enviroment.add_agent(self)  # add agent to enviroment

    def move(self):
        move_choices = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = random.choice(move_choices)  # random move, for now, will be replaced by a more sophisticated strategy
        new_x = (self.x + dx) % self.enviroment.width
        new_y = (self.y + dy) % self.enviroment.height

        self.x = new_x
        self.y = new_y

        self.visited_cells.add((self.x, self.y))
        self.update_voronoi_cell()

    def update_voronoi_cell(self):
        # need to calculate again the position of the cell
        min_dist = float('inf')
        for agent in self.enviroment.agents:
            if agent != self:
                dist = np.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
                if dist < min_dist:
                    self.current_voronoi_cell = agent
                    min_dist = dist

    def communicate_with_nearby_agents(self):
        for other_agent in self.enviroment.agents:
            if other_agent != self:
                dist = np.sqrt((self.x - other_agent.x) ** 2 + (self.y - other_agent.y) ** 2)
                if dist < 2:
                    print(f"Agente {self.id} comunica con l'agente {other_agent.id}")  # need?

    def explore(self):
        self.move()
        self.communicate_with_nearby_agents()

        # need logic to move to a certain direction

    def gather_data(self):
        # pick data, update
        self.enviroment.grid[self.x, self.y] = 1
        self.enviroment.update_map()

    def __str__(self):
        return f"Agente {self.id} in posizione ({self.x}, {self.y})"
