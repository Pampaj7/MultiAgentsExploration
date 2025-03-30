import random
import numpy as np
from dstar import initDStarLite, moveAndRescan
from utils import stateNameToCoords


class Agent:
    """
    Rappresenta un agente che si muove in un ambiente,
    esplora, comunica con altri agenti e raccoglie dati.
    """

    def __init__(self, id, enviroment, n_agents):
        """
        Inizializza un nuovo agente.

        :param id: Identificatore univoco dell'agente.
        :param x: Posizione iniziale sull'asse X.
        :param y: Posizione iniziale sull'asse Y.
        :param enviroment: Istanza dell'ambiente in cui l'agente opera.
        """
        self.id = id
        self.x = None
        self.y = None
        self.sensing_accuracy = 0.9  # Precisione del sensore dell'agente
        self.vision = 3  # Raggio di visione dell'agente
        self.enviroment = enviroment
        self.queue = []  # Coda di priorità per D* Lite
        self.k_m = 0  # Fattore di aggiustamento del costo
        self.visited_cells = {}  # Celle visitate dall'agente
        self.init_pos(n_agents)

    def init_pos(self, num_agents):

        """
        Initialize the agent's position such that agents are equally spread inside the environment.
        """
        env_width = self.enviroment.width
        env_height = self.enviroment.height

        # Calculate grid size based on the number of agents
        grid_size = int(np.sqrt(env_width * env_height / num_agents))

        # Determine the agent's position based on its id
        row = (self.id * grid_size) % env_height
        col = ((self.id * grid_size) // env_height) * grid_size % env_width

        self.x = col + random.randint(0, grid_size - 1)
        self.y = row + random.randint(0, grid_size - 1)

        # Ensure the position is within bounds
        self.x = min(self.x, env_width - 1)
        self.y = min(self.y, env_height - 1)

    def init_d_star(self):
        start_id = f'x{self.x}y{self.y}'
        goal_id = None
        frontier_points = self.enviroment.frontier_points.get(self.id, [])

        if frontier_points:
            # Use the combined heuristic to choose the best frontier point
            goal_point = self.compute_heuristic(frontier_points)
            goal_id = f'x{goal_point[0]}y{goal_point[1]}'

        self.enviroment.setStart(start_id, self.id)
        if goal_id is not None:
            self.enviroment.setGoal(goal_id, self.id)

        else:
            print(f"Agent {self.id} has no goal to reach!")
            self.enviroment.setGoal(start_id, self.id)  # Stay in place if no frontier points

        print(f"Agent {self.id} new goal: {goal_id}")  # Debugging
        self.run_d_star_lite()

    def run_d_star_lite(self):
        """ Initialize and run D* Lite for this agent """
        start_id = f'x{self.x}y{self.y}'
        goal_id = self.enviroment.goals.get(self.id)

        if goal_id:  # Ensure a goal exists before running
            self.queue = []  # Reset priority queue
            self.k_m = 0  # Reset cost adjustment factor

            print(f"Agent {self.id} running D* Lite with start {start_id} and goal {goal_id}")
            self.enviroment.resetAgentPathCosts(self.id)  # ✅ Reset costs specific to this agent

            # Initialize D* Lite and store updated queue and k_m
            _, self.queue, self.k_m = initDStarLite(self.enviroment, self.queue, start_id, goal_id, self.k_m, self.id)

    def explore(self):
        pos = f'x{self.x}y{self.y}'
        s_new, self.k_m = moveAndRescan(
            self.enviroment, self.queue, pos, self.vision, self.k_m, self.id
        )

        if s_new is None:
            print(f"⚠️ Agent {self.id} has no valid next step from {self.x, self.y} (s_new is None).")
            self.enviroment.goals[self.id] = None  # Ferma l'agente
            return

        if s_new == 'goal':
            print(f'✅ Agent {self.id} reached its goal at {self.x, self.y}!')
            self.init_d_star()
            return

        # S_new è una nuova posizione valida
        self.s_current = s_new
        self.x, self.y = stateNameToCoords(self.s_current)

        obstacle_positions = {(obs.position) for obs in self.enviroment.obstacles}
        self.visited_cells = {}
        for i in range(max(0, self.x - self.vision), min(self.enviroment.width, self.x + self.vision + 1)):
            for j in range(max(0, self.y - self.vision), min(self.enviroment.height, self.y + self.vision + 1)):
                self.visited_cells[(i, j)] = (i, j) in obstacle_positions

    def compute_heuristic(self, frontier_points):
        """
        Selects the best frontier point using a weighted combination of:
        - Distance to the agent (closer is better)
        - Density of **truly unexplored** cells (higher is better)
        """

        def distance_to_agent(point):
            """ Calculate squared Euclidean distance to the agent """
            return (point[0] - self.x) ** 2 + (point[1] - self.y) ** 2

        def density_of_frontier(point):
            """ Count **only truly unexplored (p = 0.5) cells** around the frontier point """
            x, y = point
            unexplored_count = 0
            for dx in range(-1, 2):  # Check 3x3 grid
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.enviroment.width and 0 <= ny < self.enviroment.height:
                        if self.enviroment.grid[nx, ny] == 0.5:  # Only count fully unexplored cells
                            unexplored_count += 1
            return unexplored_count

        def normalize(value, min_val, max_val):
            """ Normalize to [0,1], avoiding division by zero """
            return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

        # Step 1: Compute raw distance & density values
        distances = [distance_to_agent(point) for point in frontier_points]
        densities = [density_of_frontier(point) for point in frontier_points]

        # Step 2: Normalize distances **inverted** (so that closer points are better)
        min_distance, max_distance = min(distances), max(distances)
        normalized_distances = [1 - normalize(d, min_distance, max_distance) for d in distances]  # Inverted

        # Step 3: Normalize densities (higher is better)
        min_density, max_density = min(densities), max(densities)
        normalized_densities = [normalize(d, min_density, max_density) for d in densities]

        # Step 4: Combine scores
        distance_weight = 0.3  # Adjust based on preference
        density_weight = 0.7  # Balance between exploration and efficiency

        combined_scores = [
            distance_weight * normalized_distances[i] + density_weight * normalized_densities[i]
            for i in range(len(frontier_points))
        ]

        # Step 5: Select the best frontier
        best_frontier_index = combined_scores.index(max(combined_scores))  # Max since we inverted distance
        return frontier_points[best_frontier_index]
