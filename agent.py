import random
import numpy as np
from dstar import initDStarLite, moveAndRescan
from utils import stateNameToCoords



class Agent:
    """
    Rappresenta un agente che si muove in un ambiente,
    esplora, comunica con altri agenti e raccoglie dati.
    """

    def __init__(self, id, x, y, enviroment):
        """
        Inizializza un nuovo agente.

        :param id: Identificatore univoco dell'agente.
        :param x: Posizione iniziale sull'asse X.
        :param y: Posizione iniziale sull'asse Y.
        :param enviroment: Istanza dell'ambiente in cui l'agente opera.
        """
        self.id = id
        self.x = x
        self.y = y
        self.sensing_accuracy = 0.9  # Precisione del sensore dell'agente
        self.vision = 3 # Raggio di visione dell'agente
        self.enviroment = enviroment
        self.visited_cells = {}  # Celle visitate dall'agente 

        
    def init_d_star(self):
        start_id = f'x{self.x}y{self.y}'
        goal_id = None
        frontier_points = self.enviroment.frontier_points.get(self.id, [])
        if frontier_points:
            goal_point = random.choice(frontier_points)
            goal_id = f'x{goal_point[0]}y{goal_point[1]}'
        self.enviroment.setStart(start_id, self.id)
        if goal_id:
            self.enviroment.setGoal(goal_id, self.id)
        self.run_d_star_lite()

    def run_d_star_lite(self):
        """ Initialize and run D* Lite for this agent """
        start_id = f'x{self.x}y{self.y}'
        goal_id = self.enviroment.goals.get(self.id)

        if goal_id:  # Ensure a goal exists before running
            self.queue = []  # Reset priority queue
            self.k_m = 0  # Reset cost adjustment factor

            # Initialize D* Lite and store updated queue and k_m
            _, self.queue, self.k_m = initDStarLite(self.enviroment, self.queue, start_id, goal_id, self.k_m, self.id)
        

    def explore(self): #TODO qui c'è da metterci move and rescan

        pos = f'x{self.x}y{self.y}'
        s_new, self.k_m = moveAndRescan(
            self.enviroment, self.queue, pos, self.vision, self.k_m, self.id
        )

        if s_new == 'goal':
            print(f'Agent {self.id} reached its goal!')
        else:
            self.s_current = s_new
            pos_coords = stateNameToCoords(self.s_current)
            self.x, self.y = pos_coords  # Update agent's coordinates

        # Mark cells within vision range as visited
        self.visited_cells = {}
        for i in range(max(0, self.x - self.vision), min(self.enviroment.width, self.x + self.vision + 1)):
                for j in range(max(0, self.y - self.vision), min(self.enviroment.height, self.y + self.vision + 1)):
                    self.visited_cells[i,j] = any((i,j) == pos for pos in self.enviroment.obstacles)



