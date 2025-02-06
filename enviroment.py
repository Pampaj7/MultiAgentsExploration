import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


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

    def add_agent(self, agent):
        """
        Aggiunge un agente alla simulazione.
        """
        self.agents.append(agent)

    def update_map(self):
        """
        Aggiorna la mappa dell'ambiente in base alle celle visitate dagli agenti.
        """
        for agent in self.agents:
            for (x, y) in agent.visited_cells:
                if 0 <= x < self.width and 0 <= y < self.height:  # Controlla i limiti
                    self.grid[x, y] = 1  # Segna come esplorata

    def render(self, ax):
        """
        Visualizza la mappa dell'ambiente usando Matplotlib.
        Mostra la posizione degli agenti e le celle esplorate.
        """
        ax.clear()  # Pulisce il grafico prima di ridisegnare
        ax.imshow(self.grid, cmap='gray', origin='lower', extent=(0, self.width, 0, self.height))

        # Disegna le posizioni degli agenti
        agent_positions = np.array(
            [(agent.y, agent.x) for agent in self.agents])  # Inverti x e y per Matplotlib allucinante

        # Aggiungi un piccolo offset se necessario per allineare meglio i punti
        ax.scatter(agent_positions[:, 0] + 0.5, agent_positions[:, 1] + 0.5,
                   color='red', label='Agenti', marker='x')

        ax.set_title('Ambiente - Esplorazione')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper left')

        # Debug: stampa le posizioni degli agenti
        for i, agent in enumerate(self.agents):
            print(f"Agente {i} posizione: ({agent.x}, {agent.y})")

    def animate(self):
        """
        Esegue la visualizzazione in tempo reale della simulazione con aggiornamenti dinamici.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        def update(frame):
            """
            Funzione di aggiornamento per ogni fotogramma dell'animazione.
            """
            for agent in self.agents:
                agent.explore()  # L'agente esplora
            self.update_map()
            self.render(ax)  # Rende la mappa aggiornata
            return []

        ani = FuncAnimation(fig, update, frames=100, interval=500, blit=False)
        plt.show()

    def __str__(self):
        """
        Restituisce una rappresentazione testuale dell'ambiente.
        """
        return f"Ambiente {self.width}x{self.height} con {len(self.agents)} agenti."
