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
        self.enviroment = enviroment
        self.visited_cells = set()  # Celle visitate dall'agente
        self.current_voronoi_cell = None  # Celle di Voronoi associate

        self.enviroment.add_agent(self)  # Aggiunge l'agente all'ambiente

    def move(self):
        """
        Muove l'agente in una direzione casuale.
        """
        move_choices = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Direzioni possibili
        dx, dy = random.choice(move_choices)
        new_x = self.x + dx
        new_y = self.y + dy

        # Debug: stampa la direzione e la nuova posizione
        print(f"Agente({self.id}) si muove da ({self.y}, {self.x}) a ({new_y}, {new_x})")
        # accidenti

        # Controlla se la nuova posizione è dentro i limiti della griglia
        if 0 <= new_x < self.enviroment.width and 0 <= new_y < self.enviroment.height:
            self.x = new_x
            self.y = new_y
            self.visited_cells.add((self.x, self.y))  # Aggiungi la cella visitata
            self.update_voronoi_cell()

        else:
            # Se fuori dai limiti, non muovere l'agente
            pass

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

    def communicate_with_nearby_agents(self):
        """
        Comunica con gli agenti vicini se la distanza è inferiore a una soglia.
        """
        for other_agent in self.enviroment.agents:
            if other_agent != self:
                dist = np.sqrt((self.x - other_agent.x) ** 2 + (self.y - other_agent.y) ** 2)
                if dist < 2:
                    print(f"Agente {self.id} comunica con l'agente {other_agent.id}")

    def explore(self):
        """
        Esegue il comportamento di esplorazione:
        - Si muove in una nuova posizione.
        - Comunica con gli agenti vicini.
        """
        self.move()
        self.communicate_with_nearby_agents()

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
