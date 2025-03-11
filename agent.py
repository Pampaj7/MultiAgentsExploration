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
            # Use the combined heuristic to choose the best frontier point
            goal_point = self.compute_heuristic(frontier_points)
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
        

    def explore(self):
        pos = f'x{self.x}y{self.y}'
        s_new, self.k_m = moveAndRescan(
            self.enviroment, self.queue, pos, self.vision, self.k_m, self.id
        )

        if s_new == 'goal':
            print(f'Agent {self.id} reached its goal!')
        else:
            self.s_current = s_new
            self.x, self.y = stateNameToCoords(self.s_current)  # Update agent's coordinates

        obstacle_positions = {(obs.position) for obs in self.enviroment.obstacles}  # ✅ Now a set
        self.visited_cells ={}
        for i in range(max(0, self.x - self.vision), min(self.enviroment.width, self.x + self.vision + 1)):
            for j in range(max(0, self.y - self.vision), min(self.enviroment.height, self.y + self.vision + 1)):
                if (i, j) in obstacle_positions:
                    self.visited_cells[(i,j)]= True
                else: self.visited_cells[(i,j)]= False

    def compute_heuristic(self, frontier_points):
        """
        Combine distance to the agent and the density of unexplored cells around each frontier point.
        Returns the best frontier point based on this combined heuristic.
        """

        def distance_to_agent(point):
            # Calculate Euclidean distance squared to avoid unnecessary square roots
            return (point[0] - self.x) ** 2 + (point[1] - self.y) ** 2

        def density_of_frontier(point):
            # Calculate the density of unexplored cells around the frontier point
            x, y = point
            unexplored_count = 0
            for dx in range(-1, 2):  # Check surrounding 3x3 grid (8 neighbors)
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.enviroment.width and 0 <= ny < self.enviroment.height:
                        # Assume that the unexplored cells have a probability < 0.5
                        if self.enviroment.grid[nx, ny] < 0.5:
                            unexplored_count += 1
            return unexplored_count

        def normalize(value, min_val, max_val):
            """ Normalize value to range [0, 1] """
            return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

        # Step 1: Calculate distances and densities for all frontier points
        distances = [distance_to_agent(point) for point in frontier_points]
        densities = [density_of_frontier(point) for point in frontier_points]

        # Step 2: Normalize both metrics (we want the closer ones to have higher scores and the denser ones to have higher scores)
        min_distance = min(distances)
        max_distance = max(distances)
        normalized_distances = [normalize(d, min_distance, max_distance) for d in distances]

        min_density = min(densities)
        max_density = max(densities)
        normalized_densities = [normalize(d, min_density, max_density) for d in densities]

        # Step 3: Combine the distance and density into one heuristic value
        # You can adjust the weights to prioritize one metric over the other
        distance_weight = 0.3  # Balance factor for distance (adjustable)
        density_weight = 0.7   # Balance factor for density (adjustable)

        combined_scores = [
            distance_weight * normalized_distances[i] + density_weight * normalized_densities[i]
            for i in range(len(frontier_points))
        ]

        # Step 4: Return the frontier point with the best (lowest) combined score
        best_frontier_index = combined_scores.index(min(combined_scores))
        return frontier_points[best_frontier_index]





