import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
import matplotlib.colors as mcolors

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
        self.grid = np.zeros((width, height))  # 0 per celle libere, 1 per celle esplorate
        self.agents = []
        self.voronoi_cells = []

    def add_agent(self, agent):
        """
        Aggiunge un agente alla simulazione.
        """
        self.agents.append(agent)
        print(f"Agent added at position: ({agent.x}, {agent.y})")

    def update_map(self):
        """
        Aggiorna la mappa basandosi su una soglia adattiva per decidere se
        una cella è libera od occupata.
        """
        for agent in self.agents:
            for (x, y) in agent.visited_cells:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.grid[x, y] = min(self.grid[x, y] + 0.1, 1)  # Accumula probabilità di essere esplorata
    
    def update_voronoi(self):
        """
        Aggiorna la cella di Voronoi di ciascun agente.
        Ogni agente riceve la lista delle celle assegnate alla sua regione di Voronoi.
        """
        import numpy as np
        from scipy.spatial import Voronoi, cKDTree

        # Extract agent positions
        if len(self.agents) == 1:
            self.agents[0].current_voronoi_cell = [(x, y) for x in range(self.width) for y in range(self.height)]
            return
        if len(self.agents) == 2:
            # Calculate the perpendicular bisector,  x e y sono invertiti sennò non torna
            mid_x = (self.agents[0].y + self.agents[1].y) / 2
            mid_y = (self.agents[0].x + self.agents[1].x) / 2
            
            # Check for vertical line (same x-coordinates)
            if self.agents[0].x == self.agents[1].x:
                # If vertical, divide space based on the x-coordinate
                for x in range(self.width):
                    for y in range(self.height):
                        if x < mid_x:
                            self.agents[0].current_voronoi_cell.append((x, y))
                        else:
                            self.agents[1].current_voronoi_cell.append((x, y))
            else:
                # General case: calculate the slope of the line and its perpendicular slope
                slope = (self.agents[1].x - self.agents[0].x) / (self.agents[1].y - self.agents[0].y)
                perp_slope = -1 / slope
                
                # Assign cells based on the perpendicular bisector
                for x in range(self.width):
                    for y in range(self.height):
                        if (y - mid_y) < perp_slope * (x - mid_x):
                            self.agents[0].current_voronoi_cell.append((x, y))
                        else:
                            self.agents[1].current_voronoi_cell.append((x, y))
            return


        points = np.array([(agent.x, agent.y) for agent in self.agents])
        agent_ids = {i: agent.id for i, agent in enumerate(self.agents)}  # Map index → agent ID
        id_to_agent = {agent.id: agent for agent in self.agents}

        # Compute Voronoi diagram
        vor = Voronoi(points)

        # Create a uniform grid covering the environment
        grid_x, grid_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T  # Convert to (x, y) list

        # Find the nearest Voronoi seed (agent) for each grid cell using KDTree
        tree = cKDTree(points)
        _, nearest_agent_idx = tree.query(grid_points)  # Get index of closest agent

        # Reshape the assignment into a 2D grid
        grid_assignment = nearest_agent_idx.reshape(self.width, self.height)

        # Clear previous Voronoi cells for all agents
        for agent in self.agents:
            agent.current_voronoi_cell = []

        # Assign each grid cell to the corresponding agent **by ID**
        for x in range(self.width):
            for y in range(self.height):
                agent_index = grid_assignment[x, y]  # Find the agent index
                agent_id = agent_ids[agent_index]  # Get the agent's unique ID
                id_to_agent[agent_id].current_voronoi_cell.append((x, y))  
            
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
            for (x, y) in agent.current_voronoi_cell:
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

    def entropy(self, x, y):
        """
        Calcola l'entropia locale di una cella considerando un intorno 3x3.
        Penalizza le celle già esplorate per evitare che gli agenti vi ritornino.
        """
        kernel_size = 3  # Finestra 3x3
        half_k = kernel_size // 2
        values = []

        for dx in range(-half_k, half_k + 1):
            for dy in range(-half_k, half_k + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    values.append(self.grid[nx, ny])

        p = np.mean(values)

        # Penalizza le celle già esplorate
        if p in [0, 1]:
            return 0  # Entropia minima se completamente noto
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p) + (1 - p)  # Aggiunta penalizzazione

