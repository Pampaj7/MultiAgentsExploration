from agent import Agent
import time
import random
from enviroment import Enviroment

def main():
    # Crea un ambiente 10x10
    env = Enviroment(10, 10)

    # Aggiungi più agenti a posizioni diverse
    agent1 = Agent(1, 2, 5, env)
    agent2 = Agent(2, 3, 3, env)
    agent3 = Agent(3, 7, 7, env)

    # Aggiungi gli agenti all'ambiente
    env.add_agent(agent1)
    env.add_agent(agent2)
    env.add_agent(agent3)

    # Ogni agente farà 10 passi
    steps_per_agent = 10

    # Avvia l'animazione
    env.animate(steps=steps_per_agent)

if __name__ == "__main__":
    main()