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
        self.vision = 1 # Raggio di visione dell'agente
        self.enviroment = enviroment
        self.visited_cells = set()  # Celle visitate dall'agente
        self.current_voronoi_cell = []  # Celle di Voronoi associate

        self.enviroment.add_agent(self)  # Aggiunge l'agente all'ambiente

    def move(self):
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
                    and (nx, ny) in self.current_voronoi_cell]  # ✅ Ensure move is inside Voronoi cell
        #print(valid_moves)

        if valid_moves:
            # Find the cell with the highest entropy inside the Voronoi cell
            best_cell = None
            max_entropy = -1  

            for cell in valid_moves:
                cell_entropy = self.enviroment.entropy(cell[0], cell[1])
                if cell_entropy > max_entropy:
                    max_entropy = cell_entropy
                    best_cell = cell

            # Move to the best high-entropy cell inside the Voronoi cell
            if best_cell and max_entropy > 0:
                self.x, self.y = best_cell
            else:
                # If all entropy values are low, move randomly **inside the Voronoi region**
                self.x, self.y = random.choice(valid_moves)

        # Mark the current cell as visited
        self.visited_cells.add((self.x, self.y))
        # Add every cell inside the vision range to visited_cells
        for dx in range(-self.vision, self.vision + 1):
            for dy in range(-self.vision, self.vision + 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < self.enviroment.width and 0 <= ny < self.enviroment.height:
                    self.visited_cells.add((nx, ny))


    def update_voronoi_cell(self):
        """
        Aggiorna la cella di Voronoi dell'agente in base alla posizione corrente.
        """
        min_dist = float('inf')
        for agent in self.enviroment.agents:
            if agent != self:
                dist = np.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
                if dist < min_dist:
                    self.current_voronoi_cell = agent
                    min_dist = dist

    """     def move(self): #credo proprio che venga usata questa move TODO: togliere l'altra
   """      """
        #Muove l'agente verso la cella con maggiore entropia nell'intorno.
        #Se non ci sono celle con incertezza, si muove casualmente.
        """ """
        neighbors = [(self.x + dx, self.y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        neighbors = [(nx, ny) for nx, ny in neighbors if
                     0 <= nx < self.enviroment.width and 0 <= ny < self.enviroment.height]

        if neighbors:
            # Trova la cella con entropia massima
            best_cell = max(neighbors, key=lambda cell: self.enviroment.entropy(cell[0], cell[1]))
            self.x, self.y = best_cell
        else:
            # Se nessuna cella disponibile, muoviti casualmente
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            new_x, new_y = self.x + dx, self.y + dy
            if 0 <= new_x < self.enviroment.width and 0 <= new_y < self.enviroment.height:
                self.x, self.y = new_x, new_y

        self.visited_cells.add((self.x, self.y)) """

    def explore(self):
        """
        Esegue il comportamento di esplorazione:
        - Si muove in una nuova posizione.
        - Comunica con gli agenti vicini.
        """
        # arriva da animate, per ogni agente esegue la funzione explore
        #self.move()  # logica di movimento
        #self.communicate_with_nearby_agents()

    def gather_data(self):
        """
        Raccoglie dati dalla cella corrente e aggiorna la mappa ambientale.
        """
        self.enviroment.grid[self.x, self.y] = 1
        self.enviroment.update_map()

    def __str__(self):
        """
        Restituisce una rappresentazione testuale dell'agente.
        """
        return f"Agente {self.id} in posizione ({self.x}, {self.y})"

    def communicate_with_nearby_agents(self): #TODO: da rivedere
        """
        Controlla se un altro agente sta per muoversi nella stessa cella
        e cerca di evitare collisioni.
        """
        for other in self.enviroment.agents:
            if other != self and (self.x, self.y) == (other.x, other.y):
                # Se c'è una collisione, prova a muoverti altrove
                move_choices = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                random.shuffle(move_choices)
                for dx, dy in move_choices:
                    new_x, new_y = self.x + dx, self.y + dy
                    if 0 <= new_x < self.enviroment.width and 0 <= new_y < self.enviroment.height and (
                            new_x, new_y) not in [(a.x, a.y) for a in self.enviroment.agents]:
                        self.x, self.y = new_x, new_y
                        break
