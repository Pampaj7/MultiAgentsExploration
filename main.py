from agent import Agent
import time
import random
from enviroment import Enviroment

def main():
    # Crea un ambiente 10x10
    env = Enviroment(10, 10)

    # Aggiungi un agente alla posizione (5, 5)
    agent = Agent(1, 2, 5, env)

    # Avvia l'animazione
    env.animate()


if __name__ == "__main__":
    main()
