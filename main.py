import numpy as np
import random

from agent import Agent
from environment import Environment
from obstacle import Obstacle

# 🔒 Fix seed for reproducibility
SEED = 41
random.seed(SEED)
np.random.seed(SEED)


def main():
    # 📌 1. Creiamo un ambiente di dimensione 20x20
    env = Environment(width=20, height=20)

    # 📌 2. Creiamo 5 agenti in posizioni casuali
    num_agents = 5
    num_obstacles = 10

    for i in range(num_agents):
        agent = Agent(id=i, enviroment=env, n_agents=num_agents)
        env.add_agent(agent)

    for i in range(num_obstacles):
        while True:
            x, y = random.randint(0, env.width - 1), random.randint(0, env.height - 1)
            if not any(agent.x == x and agent.y == y for agent in env.agents):
                env.add_obstacle(Obstacle(x, y, env))
                break

    env.update_voronoi()
    env.init_env()

    for agent in env.agents:
        agent.init_d_star()

    print(f"Creato ambiente {env.width}x{env.height} con {len(env.agents)} agenti e {len(env.obstacles)} ostacoli.")

    # 📌 3. Avviamo la simulazione con 10 passi
    steps_per_agent = 120
    env.animate(steps=steps_per_agent)


if __name__ == "__main__":
    main()
