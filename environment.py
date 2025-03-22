import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import matplotlib.colors as mcolors
import random
from map_graph import Node, Graph
from utils import stateNameToCoords

matplotlib.use('TkAgg')  # oppure 'Qt5Agg', 'Qt4Agg', a seconda di ciò che hai installato


# damnit! this solved the problem

class Environment(Graph):
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
        self.frontier_points = {}
        self.graph = {}

    def init_env(self):

        self.start = {agent.id: None for agent in self.agents}
        self.goals = {agent.id: None for agent in self.agents}
        self.frontier_points = {agent.id: [] for agent in self.agents}

        pos = {obs.position for obs in self.obstacles}  # Set for fast lookup

        self.build_graph()

        for agent in self.agents:
            vision = agent.vision  # Agent's vision range
            for i in range(max(0, agent.x - vision), min(self.width, agent.x + vision + 1)):
                for j in range(max(0, agent.y - vision), min(self.height, agent.y + vision + 1)):

                    # Ensure frontier points do not touch the border of the map
                    if (
                            (i == max(0, agent.x - vision) or i == min(self.width - 1,
                                                                       agent.x + vision)) and 0 < i < self.width - 1
                    ) or (
                            (j == max(0, agent.y - vision) or j == min(self.height - 1,
                                                                       agent.y + vision)) and 0 < j < self.height - 1
                    ):
                        if (i, j) not in pos and (i, j) in self.voronoi_cells[agent.id]:
                            self.frontier_points[agent.id].append((i, j))

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
        # print(f"Obstacle added at position: ({obstacle.position})")

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
        # Initialize Voronoi cells dictionary with empty lists
        self.voronoi_cells = {agent.id: [] for agent in self.agents}

        # Case 1: Only one agent (takes all cells)
        if len(self.agents) == 1:
            self.voronoi_cells[self.agents[0].id] = [(x, y) for x in range(self.width) for y in range(self.height)]
            return

        # Case 2: Exactly two agents (use perpendicular bisector)
        if len(self.agents) == 2:
            agent1, agent2 = self.agents[0], self.agents[1]
            mid_x = (agent1.x + agent2.x) / 2
            mid_y = (agent1.y + agent2.y) / 2

            # Vertical bisector (same x-coordinate)
            if agent1.y == agent2.y:
                first_agent = agent1 if agent1.x < agent2.x else agent2
                second_agent = agent1 if agent1.x > agent2.x else agent2
                for x in range(self.width):
                    for y in range(self.height):
                        if x < mid_x:
                            self.voronoi_cells[first_agent.id].append((x, y))
                        else:
                            self.voronoi_cells[second_agent.id].append((x, y))

            # Horizontal bisector (same y-coordinate)
            elif agent1.x == agent2.x:
                first_agent = agent1 if agent1.y < agent2.y else agent2
                second_agent = agent1 if agent1.y > agent2.y else agent2
                for x in range(self.width):
                    for y in range(self.height):
                        if y < mid_y:
                            self.voronoi_cells[first_agent.id].append((x, y))
                        else:
                            self.voronoi_cells[first_agent.id].append((x, y))

            # General case: Perpendicular bisector
            else:
                for x in range(self.width):
                    for y in range(self.height):
                        dist1 = (x - agent1.x) ** 2 + (y - agent1.y) ** 2  # Squared Euclidean distance to agent1
                        dist2 = (x - agent2.x) ** 2 + (y - agent2.y) ** 2  # Squared Euclidean distance to agent2

                        if dist1 < dist2:
                            self.voronoi_cells[agent1.id].append((x, y))
                        else:
                            self.voronoi_cells[agent2.id].append((x, y))
            return

        # Case 3: More than two agents (Voronoi diagram)
        points = np.array([(agent.x, agent.y) for agent in self.agents])
        index_to_agent = {i: agent for i, agent in enumerate(self.agents)}

        # Voronoi diagram and KDTree
        vor = Voronoi(points)
        grid_x, grid_y = np.meshgrid(np.arange(self.width), np.arange(self.height),
                                     indexing="ij")  # Ensure correct indexing
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T  # Keep (x, y) format
        tree = cKDTree(points)
        _, nearest_agent_idx = tree.query(grid_points)
        grid_assignment = nearest_agent_idx.reshape(self.width, self.height)

        # Assign cells to each agent
        for x in range(self.width):
            for y in range(self.height):
                agent_index = grid_assignment[x, y]
                self.voronoi_cells[index_to_agent[agent_index].id].append((x, y))

    def update_frontier(self):
        """
        Update the frontier points of the environment.
        """
        new_frontier = {agent.id: [] for agent in self.agents}  # Initialize empty lists

        for agent in self.agents:
            vision = agent.vision  # Agent's vision range
            voronoi_cells = self.voronoi_cells[agent.id]  # Get agent's Voronoi cells

            for (x, y) in voronoi_cells:  # Only check cells inside the agent's Voronoi region
                if self.grid[x, y] >= 0.5:  # Unexplored cells are not frontiers themselves
                    continue

                # Exclude map borders
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    continue  # Skip border cells

                # Check neighbors to see if this is a frontier point
                neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                for nx, ny in neighbors:
                    if 0 < nx < self.width and 0 < ny < self.height:
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

    def render(self, ax, show_cell_coords=True, show_probabilities=False, show_legend=False):
        """
        Visualizza la mappa dell'ambiente usando Matplotlib.
        Mostra la posizione degli agenti, ostacoli, celle Voronoi, frontiere e goal.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from utils import stateNameToCoords

        fig = plt.gcf()
        fig.set_size_inches(12, 12)
        ax.clear()

        # 1. Mappa base (probabilità di occupazione)
        cmap = plt.get_cmap('gray_r')
        norm = mcolors.Normalize(vmin=0, vmax=1)
        ax.imshow(self.grid, cmap=cmap, norm=norm, origin='lower',
                  extent=(0, self.grid.shape[1], 0, self.grid.shape[0]))

        # 2. Coordinate o probabilità dentro le celle
        for row in range(self.height):
            for col in range(self.width):
                if show_cell_coords:
                    ax.text(col + 0.5, row + 0.5, f'({col},{row})', color='black',
                            ha='center', va='center', fontsize=8)
                elif show_probabilities:
                    ax.text(col + 0.5, row + 0.5, f'{self.grid[row, col]:.2f}', color='black',
                            ha='center', va='center', fontsize=8)

        # 3. Regioni di Voronoi (non trasposte, disegnate con row, col)
        cmap_voronoi = plt.get_cmap("tab10")
        norm_voronoi = mcolors.Normalize(vmin=0, vmax=max(len(self.agents) - 1, 1))
        voronoi_grid = np.full((self.height, self.width), -1)
        for agent_id, cells in self.voronoi_cells.items():
            for (x, y) in cells:  # x=row, y=col
                voronoi_grid[x, y] = agent_id
        ax.pcolormesh(np.arange(self.width + 1), np.arange(self.height + 1),
                      voronoi_grid, cmap=cmap_voronoi, norm=norm_voronoi, alpha=0.4)

        # 4. Posizioni agenti
        agent_positions = np.array([(agent.x, agent.y) for agent in self.agents])
        ax.scatter(agent_positions[:, 1] + 0.5, agent_positions[:, 0] + 0.5,
                   color='red', label='Agenti', marker='x', s=100)

        # 5. Ostacoli
        for obs in self.obstacles:
            x, y = obs.position  # x=row, y=col
            ax.scatter(y + 0.5, x + 0.5, color='purple', marker='s', s=50)
            ax.text(y + 0.5, x + 0.5, f'({y},{x})', color='white', ha='center',
                    va='center', fontsize=8)

        # 6. Punti di frontiera
        for agent_id, points in self.frontier_points.items():
            if points:
                frontier_positions = np.array(points)
                if frontier_positions.ndim == 2 and frontier_positions.shape[1] == 2:
                    ax.scatter(frontier_positions[:, 1] + 0.5,
                               frontier_positions[:, 0] + 0.5,
                               color='blue', label=f'Frontiera {agent_id}',
                               marker='o', s=50)

        # 7. Goal per agente
        for agent_id, goal in self.goals.items():
            if goal is not None:
                gx, gy = stateNameToCoords(goal)  # gx=row, gy=col
                ax.scatter(gy + 0.5, gx + 0.5, color='orange', marker='x', s=100,
                           label=f'Goal {agent_id}')

        ax.set_title('Ambiente - Esplorazione con Voronoi')
        ax.set_xlabel('X (Colonne)')
        ax.set_ylabel('Y (Righe)')

        if show_legend:
            ax.legend(loc='upper right')

    def animate(self, steps, interval=1000):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        def update(frame):
            ax.clear()
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)

            for agent in self.agents:
                agent.explore()

            self.update_map()
            self.update_graph()
            self.update_frontier()
            self.render(ax)

            return []

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

                possible_neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

                # Check if the point is inside the agent's own Voronoi cell
                if (x, y) in self.voronoi_cells[agent.id]:
                    owner_id = agent.id  # The agent owns this cell
                else:
                    # Find which agent owns this cell
                    owner_id = None
                    for other_agent in self.agents:
                        if (x, y) in self.voronoi_cells[other_agent.id]:
                            owner_id = other_agent.id
                            break  # Stop once the owner is found

                # Ensure that the node is correctly added to the owner’s graph
                if owner_id is not None:
                    neighbors = [
                        f'x{n[0]}y{n[1]}'
                        for n in possible_neighbors
                        if 0 <= n[0] < self.width and 0 <= n[1] < self.height and self.grid[n[0]][n[1]] != 0.5 and (
                            n[0], n[1]) in self.voronoi_cells[owner_id]
                    ]

                    # ✅ Add the node to the correct agent's graph
                    if owner_id in self.graph:  # Ensure the agent has a graph
                        self.addNodeToGraph(id, neighbors, owner_id)
                        # print(f"Added node {id} to agent {owner_id}'s graph with neighbors {neighbors}")

    def build_graph(self):
        edge = 1
        self.graph = {agent.id: {} for agent in self.agents}
        for agent in self.agents:
            vision = agent.vision  # Agent's vision range

            for i in range(max(0, agent.x - vision), min(self.width, agent.x + vision + 1)):
                for j in range(max(0, agent.y - vision), min(self.height, agent.y + vision + 1)):
                    node_id = f'x{i}y{j}'

                    # If node does not exist, create it
                    if node_id not in self.graph[agent.id]:
                        self.graph[agent.id][node_id] = Node(node_id)

                    node = self.graph[agent.id][node_id]  # Get the node reference

                    # Check and connect with existing neighbors
                    neighbors = [
                        (i - 1, j),  # Top
                        (i + 1, j),  # Bottom
                        (i, j - 1),  # Left
                        (i, j + 1)  # Right
                    ]
                    for ni, nj in neighbors:
                        neighbor_id = f'x{ni}y{nj}'
                        if 0 <= ni < self.height and 0 <= nj < self.width:  # Ensure it's in bounds
                            if neighbor_id in self.graph[agent.id]:  # If neighbor already exists, connect them
                                node.parents[neighbor_id] = edge
                                node.children[neighbor_id] = edge
                                self.graph[agent.id][neighbor_id].parents[node_id] = edge
                                self.graph[agent.id][neighbor_id].children[node_id] = edge

    def resetAgentPathCosts(self, agent_id):
        """Reset path information (rhs, g, and other values) only for the specific agent"""

        # Reset the rhs, g values and queue entries for the current agent's start and goal
        goal_id = self.goals[agent_id]
        for node in self.graph[agent_id].values():
            node.g = float('inf')
            node.rhs = float('inf')

        self.graph[agent_id][goal_id].rhs = 0  # The goal always has rhs 0 at the start
