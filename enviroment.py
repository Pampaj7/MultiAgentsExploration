import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import matplotlib.colors as mcolors
import random
from map_graph import Node,Graph
from utils import stateNameToCoords


matplotlib.use('TkAgg')  # oppure 'Qt5Agg', 'Qt4Agg', a seconda di ciò che hai installato


# damnit! this solved the problem

class Enviroment(Graph):
    """
    Classe che rappresenta un ambiente bidimensionale esplorato da agenti.
    """

    def __init__(self, width, height):
        """
        Inizializza l'ambiente con dimensioni specificate.
        """
        self.width = width
        self.height = height
        self.grid = np.full((self.width, self.height), 0.5) 
        self.agents = []
        self.obstacles = []
        self.voronoi_cells = {}
        self.frontier_points= {}
        self.graph = {}

    def init_env(self):
        self.start = {agent.id: None for agent in self.agents}
        self.goals = {agent.id: None for agent in self.agents}
        self.frontier_points = {agent.id: [] for agent in self.agents}
        for agent in self.agents:
            vision = agent.vision  # Agent's vision range
            for i in range(max(0, agent.x - vision), min(self.width, agent.x + vision + 1)):
                for j in range(max(0, agent.y - vision), min(self.height, agent.y + vision + 1)):
                    if i == max(0, agent.x - vision) or i == min(self.width, agent.x + vision) or j == max(0, agent.y - vision) or j == min(self.height, agent.y + vision):
                        self.frontier_points[agent.id].append((i, j))
        pos = {obs.position for obs in self.obstacles}  # Set for fast lookup

        for agent in self.agents:
            for i in range(max(0, agent.x - agent.vision), min(self.width, agent.x + agent.vision + 1)):
                for j in range(max(0, agent.y - agent.vision), min(self.height, agent.y + agent.vision + 1)):
                    p_obs_given_occupied, p_obs_given_free = (
                        (agent.sensing_accuracy, 1 - agent.sensing_accuracy) if (i, j) in pos 
                        else (1 - agent.sensing_accuracy, agent.sensing_accuracy)
                    )
                    self.grid[i, j] = self.bayes_update(self.grid[i, j], p_obs_given_occupied, p_obs_given_free)

    def add_agent(self, agent):
        """
        Aggiunge un agente alla simulazione.
        """
        self.agents.append(agent)
        print(f"Agent added at position: ({agent.x}, {agent.y})")

    def add_obstacle(self, obstacle):
        """
        Aggiunge un ostacolo all'ambiente.
        """
        self.obstacles.append(obstacle)
        print(f"Obstacle added at position: ({obstacle.position})")

    def update_map(self):
        """
        Aggiorna la mappa basandosi su una soglia adattiva per decidere se
        una cella è libera od occupata.
        utilizza le probabilità accumulate per ogni cella.
        """
        for agent in self.agents:
            for (x, y), occupied in agent.visited_cells.items():  # Directly iterate over visited cells
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Compute probabilities explicitly
                    if occupied:
                        p_obs_given_occupied = agent.sensing_accuracy
                        p_obs_given_free = 1 - agent.sensing_accuracy
                    else:
                        p_obs_given_occupied = 1 - agent.sensing_accuracy
                        p_obs_given_free = agent.sensing_accuracy

                    # Bayesian update
                    self.grid[x, y] = self.bayes_update(self.grid[x, y], p_obs_given_occupied, p_obs_given_free)


        for x, y in np.ndindex(self.grid.shape):
            if (x, y) not in agent.visited_cells:
                if self.grid[x, y] < 0.5:
                    self.grid[x, y] += 0.01
                elif self.grid[x, y] > 0.5:
                    self.grid[x, y] -= 0.01
                    
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
                if self.agents[1].y != self.agents[0].y:
                    slope = (self.agents[1].x - self.agents[0].x) / (self.agents[1].y - self.agents[0].y)
                else:
                    slope = float('inf')  # Use infinity to represent a vertical line
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

    def update_frontier(self):
        """
        Update the frontier points of the environment.
        """
   
        new_frontier = {agent.id: [] for agent in self.agents}  # Initialize empty lists

        for agent in self.agents:
            vision = agent.vision  # Agent's vision range
            voronoi_cells = self.voronoi_cells[agent.id]  # Get agent's Voronoi cells

            for x, y in voronoi_cells:  # Only check cells inside the agent's Voronoi region
                if self.grid[x, y] == 0.5:  # Unexplored cells are not frontiers themselves
                    continue

                # Check neighbors to see if this is a frontier point
                neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                for nx, ny in neighbors:
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.grid[nx, ny] == 0.5:  # Unknown neighbor found
                            new_frontier[agent.id].append((x, y))
                            break  # No need to check other neighbors

        self.frontier_points = new_frontier  # Update the stored frontier points

    def update_obstacles(self):
        """
        Update the obstacles in the environment.
        """
        for obs in self.obstacles:
            obs.move()      

    def render(self, ax):
        """
        Visualizza la mappa dell'ambiente usando Matplotlib.
        Mostra la posizione degli agenti e le celle esplorate, evidenziando le celle Voronoi.
        """
        ax.clear()  # Pulisce il grafico prima di ridisegnare

        # Visualizza la mappa di base con sfumature di grigio invertite in base alla probabilità
        cmap = plt.get_cmap('gray_r')  # Usa una colormap in scala di grigi invertita
        norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalizza tra 0 e 1
        ax.imshow(self.grid, cmap=cmap, norm=norm, origin='lower', extent=(0, self.width, 0, self.height))

        # Definisci una colormap per le regioni di Voronoi
        cmap_voronoi = plt.get_cmap("tab10")  # Usa 10 colori diversi per gli agenti
        norm_voronoi = mcolors.Normalize(vmin=0, vmax=len(self.agents) - 1)  # Normalizza ID agenti

        # Disegna le regioni di Voronoi
        voronoi_grid = np.full((self.width, self.height), -1)  # Inizializza griglia con -1 (nessuna regione)
        for agent_id, cells in self.voronoi_cells.items():
            for (x, y) in cells:
                voronoi_grid[x, y] = agent_id  # Swap x, y

        # Usa pcolormesh per colorare le regioni Voronoi
        ax.pcolormesh(np.arange(self.width + 1), np.arange(self.height + 1), voronoi_grid.T,
                    cmap=cmap_voronoi, norm=norm_voronoi, alpha=0.4)

        # Disegna le posizioni degli agenti
        agent_positions = np.array([(agent.x, agent.y) for agent in self.agents])  # Correct (x, y) order
        ax.scatter(agent_positions[:, 1] + 0.5, agent_positions[:, 0] + 0.5,
                color='red', label='Agenti', marker='x', s=100)

        # Stampa la probabilità in ogni cella
        # for x in range(self.width):
        #     for y in range(self.height):
        #         ax.text(y + 0.5, x + 0.5, f'{self.grid[x, y]:.2f}', 
        #                 color='black', ha='center', va='center', fontsize=8)

        # Disegna i punti di frontiera per ogni agente
        for agent_id, points in self.frontier_points.items():
            if points:  # Ensure there are points to plot
                frontier_positions = np.array(points)  # Convert to NumPy array
                if frontier_positions.ndim == 2 and frontier_positions.shape[1] == 2:
                    ax.scatter(frontier_positions[:, 1] + 0.5,  # X-coordinates
                            frontier_positions[:, 0] + 0.5,  # Y-coordinates
                            color='blue',
                            label=f'Frontiera Agente {agent_id}',
                            marker='o',
                            s=50)
        
        # Disegna gli ostacoli  
        for obs in self.obstacles:
            ax.scatter(obs.position[1] + 0.5, obs.position[0] + 0.5, color='purple', marker='s', s=50)
        
        # Disegna il Goal
        for agent_id, goal in self.goals.items():
            if goal is not None:
                goal_coords = stateNameToCoords(goal)
                ax.scatter(goal_coords[1] + 0.5, goal_coords[0] + 0.5, color='orange', marker='x', s=100, label=f'Goal Agente {agent_id}')



        ax.set_title('Ambiente - Esplorazione con Voronoi')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        #ax.legend(loc='upper left')

    def animate(self, steps=10, interval=1000):
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
                agent.explore()  # Ogni agente esplora (fa un passo)

            self.update_map() 
            #self.update_graph()
            self.update_frontier()  
            #self.update_obstacles() 
            self.render(ax)  # Rende la mappa aggiornata
            return []

        # 10 è il numero di passi totali per tutti gli agenti
        ani = FuncAnimation(fig, update, frames=steps, interval=interval)
        plt.show()

    def __str__(self):
        """
        Restituisce una rappresentazione testuale dell'ambiente.
        """
        return f"Ambiente {self.width}x{self.height} con {len(self.agents)} agenti."

    def update_graph(self):
        for agent in self.agents:  # Loop through all agents
            for (x, y), occupied in agent.visited_cells.items():  # Loop through all visited cells
                
                id = f'x{x}y{y}'  # Convert to string ID

                # Find neighbors (4-connected grid: up, down, left, right)
                neighbors = []
                possible_neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                
                neighbors = [f'x{n[0]}y{n[1]}' for n in possible_neighbors if self.grid[n[0]][n[1]] != 0.5]
                # Call addNodeToGraph to add/update the node in the graph
                self.addNodeToGraph(id, neighbors)

    def build_graph(self):
        edge = 1
        for agent in self.agents:
            vision = agent.vision  # Agent's vision range

            for i in range(max(0, agent.x - vision), min(self.width, agent.x + vision + 1)):
                for j in range(max(0, agent.y - vision), min(self.height, agent.y + vision + 1)):
                    node_id = f'x{i}y{j}'

                    # If node does not exist, create it
                    if node_id not in self.graph:
                        self.graph[node_id] = Node(node_id)

                    node = self.graph[node_id]  # Get the node reference

                    # Check and connect with existing neighbors
                    neighbors = [
                        (i - 1, j),  # Top
                        (i + 1, j),  # Bottom
                        (i, j - 1),  # Left
                        (i, j + 1)   # Right
                    ]
                    for ni, nj in neighbors:
                        neighbor_id = f'x{ni}y{nj}'
                        if 0 <= ni < self.height and 0 <= nj < self.width:  # Ensure it's in bounds
                            if neighbor_id in self.graph:  # If neighbor already exists, connect them
                                node.parents[neighbor_id] = edge
                                node.children[neighbor_id] = edge
                                self.graph[neighbor_id].parents[node_id] = edge
                                self.graph[neighbor_id].children[node_id] = edge

    


        
