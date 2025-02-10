from agent import Agent
from enviroment import Enviroment
import random


def main():
    # 📌 1. Creiamo un ambiente di dimensione 20x20
    env = Enviroment(width=20, height=20)

    # 📌 2. Creiamo 5 agenti in posizioni casuali
    num_agents = 5
    agents = []
    for i in range(num_agents):
        x, y = random.randint(0, env.width - 1), random.randint(0, env.height - 1)
        agent = Agent(id=i, x=x, y=y, enviroment=env)
        agents.append(agent)

    print(f"Creato ambiente {env.width}x{env.height} con {num_agents} agenti.")

    # 📌 3. Avviamo la simulazione con 50 passi
    steps_per_agent = 50
    env.animate(steps=steps_per_agent)


if __name__ == "__main__":
    main()
