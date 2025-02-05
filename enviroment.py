import matplotlib.pyplot as plt
import numpy as np
import random

from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, voronoi_plot_2d


class Enviroment:
    def __init__(self, width, height):

        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))  # 0 free 1 occ
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def update_map(self):
        self.grid.fill(0)
        for agent in self.agents:
            for (x, y) in agent.visited_cells:
                self.grid[x, y] = 1

    def render(self):
        """
        Visualizza la mappa dell'ambiente usando Matplotlib.
        Mostra la posizione degli agenti e le celle esplorate.
        """
        plt.imshow(self.grid, cmap='gray', origin='lower', extent=(0, self.width, 0, self.height))

        agent_positions = np.array([(agent.x, agent.y) for agent in self.agents])
        plt.scatter(agent_positions[:, 0], agent_positions[:, 1], color='red', label='Agenti', marker='x')

        plt.title('Ambiente - Esplorazione')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='upper left')
        plt.show()

    def animate(self):
        """
        Esegue la visualizzazione in tempo reale della simulazione.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        # Funzione di aggiornamento per ogni fotogramma dell'animazione
        def update(frame):
            ax.clear()
            self.update_map()
            self.render()
            for agent in self.agents:
                agent.explore()  # L'agente esplora nell'animazione
            return []

        ani = FuncAnimation(fig, update, frames=100, interval=500)
        plt.show()

    def __str__(self):
        return f"Ambiente {self.width}x{self.height} con {len(self.agents)} agenti."
