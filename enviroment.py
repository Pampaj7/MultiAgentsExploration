import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import matplotlib.colors as mcolors
import random

matplotlib.use('TkAgg')  # oppure 'Qt5Agg', 'Qt4Agg', a seconda di ciò che hai installato


# damnit! this solved the problem

class Enviroment:
    """
    Classe che rappresenta un ambiente bidimensionale esplorato da agenti.
    """

    def __init__(self, width, height):
        """
        Inizializza l'ambiente con dimensioni specificate.
        """
        self.width = width
        self.height = height
        self.grid = None # contiene la probabilità di essere occupata
        self.agents = []
        self.obstacles = []
        self.voronoi_cells = {}
        self.frontier_points= {}
        self.init()

    def init(self):
        self.grid = np.full((self.width, self.height), 0.5)  
        for i in range(5):
            while True:
                x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
                if (x, y) not in self.obstacles and all(agent.x != x or agent.y != y for agent in self.agents):
                    self.obstacles.append((x, y))
                    break
        


    def add_agent(self, agent):
        """
        Aggiunge un agente alla simulazione.
        """
        self.agents.append(agent)
        print(f"Agent added at position: ({agent.x}, {agent.y})")

    def add_obstacle(self, x, y):
        """
        Aggiunge un ostacolo all'ambiente.
        """
        self.obstacles.append((x, y))

    def update_map(self):
        """
        Aggiorna la mappa basandosi su una soglia adattiva per decidere se
        una cella è libera od occupata.
        utilizza le probabilità accumulate per ogni cella.
        """
        for agent in self.agents:
            for (x, y), occupancy in agent.visited_cells.items():  # Get both coordinates and occupancy
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Determine sensor likelihoods based on observation
                    p_obs_given_occupied = agent.sensing_accuracy if occupancy == 1 else 1 - agent.sensing_accuracy
                    p_obs_given_free = 1 - agent.sensing_accuracy if occupancy == 1 else agent.sensing_accuracy

                    # Bayesian update and store result
                    self.grid[x, y] = self.bayes_update(self.grid[x, y], p_obs_given_occupied, p_obs_given_free)


    def bayes_update(self, prior, p_obs_given_occupied, p_obs_given_free):
        """
        Perform a Bayesian update of occupancy probability.

        Args:
            prior (float): Prior probability of the cell being occupied
            p_obs_given_occupied (float): Probability of the observation given the cell is occupied.
            p_obs_given_free (float): Probability of the observation given the cell is free.

        Returns:
            float: Updated posterior probability of the cell being occupied.
        """        
        evidence = p_obs_given_occupied * prior + p_obs_given_free * (1 - prior)
        posterior = (p_obs_given_occupied * prior) / evidence if evidence > 0 else prior
        return posterior
    
    def update_voronoi(self):
        """
        Update the Voronoi cells of each agent.
        Each agent receives the list of cells assigned to its Voronoi region.
        """
        import numpy as np
        from scipy.spatial import Voronoi, cKDTree

        # Initialize Voronoi cells dictionary with empty lists
        self.voronoi_cells = {agent.id: [] for agent in self.agents}  # 🟢 FIXED HERE

        # Case 1: Only one agent (takes all cells)
        if len(self.agents) == 1:
            self.voronoi_cells[self.agents[0].id] = [(x, y) for x in range(self.width) for y in range(self.height)]
            return

        # Case 2: Exactly two agents (use perpendicular bisector)
        if len(self.agents) == 2:
            # Midpoint
            mid_x = (self.agents[0].y + self.agents[1].y) / 2
            mid_y = (self.agents[0].x + self.agents[1].x) / 2

            # If vertical line (equal x-coordinates)
            if self.agents[0].x == self.agents[1].x:
                for x in range(self.width):
                    for y in range(self.height):
                        if x < mid_x:
                            self.voronoi_cells[self.agents[0].id].append((x, y))
                        else:
                            self.voronoi_cells[self.agents[1].id].append((x, y))
            else:
                # General case: calculate slope and perpendicular slope
                slope = (self.agents[1].x - self.agents[0].x) / (self.agents[1].y - self.agents[0].y)
                perp_slope = -1 / slope

                for x in range(self.width):
                    for y in range(self.height):
                        if (y - mid_y) < perp_slope * (x - mid_x):
                            self.voronoi_cells[self.agents[0].id].append((x, y))
                        else:
                            self.voronoi_cells[self.agents[1].id].append((x, y))
            return

        # Case 3: More than two agents (Voronoi diagram)
        points = np.array([(agent.x, agent.y) for agent in self.agents])
        index_to_agent = {i: agent for i, agent in enumerate(self.agents)}

        # Voronoi diagram and KDTree
        vor = Voronoi(points)
        grid_x, grid_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        tree = cKDTree(points)
        _, nearest_agent_idx = tree.query(grid_points)
        grid_assignment = nearest_agent_idx.reshape(self.width, self.height)

        # Assign cells to each agent
        for x in range(self.width):
            for y in range(self.height):
                agent_index = grid_assignment[x, y]
                self.voronoi_cells[index_to_agent[agent_index].id].append((x, y))  # ✅ FIXED HERE


            
    def render(self, ax):
        """
        Visualizza la mappa dell'ambiente usando Matplotlib.
        Mostra la posizione degli agenti e le celle esplorate, evidenziando le celle Voronoi.
        """
        ax.clear()  # Pulisce il grafico prima di ridisegnare
        
        # Visualizza la mappa di base in grigio
        ax.imshow(self.grid, cmap='gray', origin='lower', extent=(0, self.width, 0, self.height))

        # Definisci una colormap per le regioni di Voronoi
        cmap = plt.get_cmap("tab10")  # Usa 10 colori diversi per gli agenti
        norm = mcolors.Normalize(vmin=0, vmax=len(self.agents) - 1)  # Normalizza ID agenti

        # Disegna le regioni di Voronoi
        voronoi_grid = np.full((self.width, self.height), -1)  # Inizializza griglia con -1 (nessuna regione)

        for agent in self.agents:
            for (x, y) in self.voronoi_cells[agent.id]:
                voronoi_grid[x, y] = agent.id  # Assegna ID dell'agente alla cella
        
        # Usa pcolormesh per colorare le regioni Voronoi
        ax.pcolormesh(np.arange(self.width + 1), np.arange(self.height + 1), voronoi_grid.T,
                    cmap=cmap, norm=norm, alpha=0.4)  # `alpha=0.4` rende le regioni semitrasparenti

        # Disegna le posizioni degli agenti
        agent_positions = np.array([(agent.y, agent.x) for agent in self.agents])  # Inverti x e y per Matplotlib
        
        ax.scatter(agent_positions[:, 0] + 0.5, agent_positions[:, 1] + 0.5,
                color='red', label='Agenti', marker='x', s=100)

        ax.set_title('Ambiente - Esplorazione con Voronoi')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper left')


    def animate(self, steps=10):
        # funziona partenza
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        def update(frame):
            """
            Funzione di aggiornamento per ogni fotogramma dell'animazione.
            """
            # Ogni agente esegue un movimento per ogni fotogramma
            for agent in self.agents:
                # per ogni agente lanciamo la funzione di explore ctlr+b to go
                agent.explore()  # Ogni agente esplora (fa un passo)

            self.update_map()
            print(self.grid)
            print("_______________FINE GRID____________________")
            self.render(ax)  # Rende la mappa aggiornata
            return []

        # 10 è il numero di passi totali per tutti gli agenti
        ani = FuncAnimation(fig, update, frames=steps, interval=1000 / 3)
        plt.show()

    def __str__(self):
        """
        Restituisce una rappresentazione testuale dell'ambiente.
        """
        return f"Ambiente {self.width}x{self.height} con {len(self.agents)} agenti."

