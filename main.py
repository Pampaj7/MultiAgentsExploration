from agent import Agent
from enviroment import Enviroment
import random
from obstacle import Obstacle


def main():
    # 📌 1. Creiamo un ambiente di dimensione 20x20
    env = Enviroment(width=20, height=20)

    # 📌 2. Creiamo 5 agenti in posizioni casuali
    num_agents = 1
    num_obstacles = 30

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

    # 📌 3. Avviamo la simulazione con 50 passi
    steps_per_agent = 50
    env.animate(steps=steps_per_agent)


if __name__ == "__main__":
    main()
