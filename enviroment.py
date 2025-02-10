import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

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

    def add_agent(self, agent):
        """
        Aggiunge un agente alla simulazione.
        """
        self.agents.append(agent)

    def update_map(self):
        """
        Aggiorna la mappa basandosi su una soglia adattiva per decidere se
        una cella è libera od occupata.
        """
        for agent in self.agents:
            for (x, y) in agent.visited_cells:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.grid[x, y] = min(self.grid[x, y] + 0.1, 1)  # Accumula probabilità di essere esplorata

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

