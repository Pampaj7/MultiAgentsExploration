import random
import numpy as np


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
         # Celle di Voronoi associate #TODO da togliere

        self.enviroment.add_agent(self)  # Aggiunge l'agente all'ambiente

    def move(self):#TODO: da rivedere con entropia cambiata
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


    def explore(self):
        """
        Esegue il comportamento di esplorazione:
        - Si muove in una nuova posizione.
        - Comunica con gli agenti vicini.
        """
        # arriva da animate, per ogni agente esegue la funzione explore
        self.move()  # logica di movimento
        



