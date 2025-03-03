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
        self.vision = 1 # Raggio di visione dell'agente
        self.enviroment = enviroment
        self.visited_cells = {}  # Celle visitate dall'agente 
        self.init()
        
    def init(self):

        start_id = f'x{self.x}y{self.y}'
        goal_id = None
        frontier_points = self.enviroment.frontier_points[self.id]
        if frontier_points:
            goal_point = random.choice(frontier_points)
            goal_id = f'x{goal_point[0]}y{goal_point[1]}'
        self.enviroment.setStart(start_id, self.id)
        if goal_id : self.enviroment.setGoal(goal_id, self.id)
        self.runDStarLite()

    def runDStarLite(self):
        """ Initialize and run D* Lite for this agent """
        start_id = f'x{self.x}y{self.y}'
        goal_id = self.enviroment.goals.get(self.id)

        if goal_id:  # Ensure a goal exists before running
            self.queue = []  # Reset priority queue
            self.k_m = 0  # Reset cost adjustment factor

            # Initialize D* Lite and store updated queue and k_m
            _, self.queue, self.k_m = initDStarLite(self.enviroment, self.queue, start_id, goal_id, self.k_m)
        

    def move(self):#TODO: da rivedere con entropia cambiata da togliere proprio dato che il movimento è dato dal move and rescan
        """
        Moves the agent towards the cell with the highest entropy **inside its Voronoi cell**.
        If all nearby cells have low entropy, the agent moves randomly within its Voronoi region.
        """
        # Define all possible moves in a 3×3 neighborhood
        possible_moves = [
            (self.x - 1, self.y),  # Up
            (self.x + 1, self.y),  # Down
            (self.x, self.y - 1),  # Left
            (self.x, self.y + 1),  # Right
            (self.x - 1, self.y - 1),  # Up-Left
            (self.x - 1, self.y + 1),  # Up-Right
            (self.x + 1, self.y - 1),  # Down-Left
            (self.x + 1, self.y + 1)  # Down-Right
        ]

        # Filter moves that are inside both the grid AND the agent's Voronoi region
        valid_moves = [(nx, ny) for nx, ny in possible_moves
                    if (0 <= nx < self.enviroment.width and 0 <= ny < self.enviroment.height) 
                    and (nx, ny) in self.enviroment.voronoi_cells[self.id]]  # ✅ Ensure move is inside Voronoi cell
        #print(valid_moves)

        if valid_moves:
            # Find the cell with the highest entropy inside the Voronoi cell
            best_cell = None
            max_entropy = -1  

            for cell in valid_moves:
                cell_entropy = self.entropy(cell[0], cell[1])
                if cell_entropy > max_entropy:
                    max_entropy = cell_entropy
                    best_cell = cell

            # Move to the best high-entropy cell inside the Voronoi cell
            if best_cell and max_entropy > 0:
                self.x, self.y = best_cell
            else:
                # If all entropy values are low, move randomly **inside the Voronoi region**
                self.x, self.y = random.choice(valid_moves)

        self.visited_cells = {} 
        # Add every cell inside the vision range to visited_cells TODO: fare una funzione sensing da richiamare, forse è più pulito
        for dx in range(-self.vision, self.vision + 1):
            for dy in range(-self.vision, self.vision + 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < self.enviroment.width and 0 <= ny < self.enviroment.height:
                    self.visited_cells[(nx, ny)] = 1 if (nx, ny) in [obs.position for obs in self.enviroment.obstacles] else 0

    def entropy(self, x, y):
        """
        Calcola l'entropia massima all'interno della cella di Voronoi dell'agente.
        """
        kernel_size = 3  # Finestra 3x3
        half_k = kernel_size // 2
        values = []

        for dx in range(-half_k, half_k + 1):
            for dy in range(-half_k, half_k + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.enviroment.width and 0 <= ny < self.enviroment.height:
                    values.append(self.enviroment.grid[nx, ny])

        p = np.mean(values)

        # Penalizza le celle già esplorate
        if p in [0, 1]:
            return 0  # Entropia minima se completamente noto
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p) + (1 - p)  # Aggiunta penalizzazione


    def explore(self): #TODO qui c'è da metterci move and rescan


        s_new, self.k_m = moveAndRescan(
            self.enviroment, self.queue, self.s_current, self.vision, self.k_m
        )

        if s_new == 'goal':
            print(f'Agent {self.id} reached its goal!')
        else:
            self.s_current = s_new
            pos_coords = stateNameToCoords(self.s_current)
            self.x, self.y = pos_coords  # Update agent's coordinates



