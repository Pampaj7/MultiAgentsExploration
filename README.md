# Multi-Agent Exploration

## Description
This project implements a simulation of **multi-agent exploration**, using:
- **Voronoi cells** to divide the environment spatially.
- **Entropy** to guide agents toward less explored areas.
- A **probabilistic map update** strategy with a threshold to classify free vs occupied cells.
- **Shortest-path algorithms** (A*, BFS) for efficient navigation.
- **Real-time visualization** using Matplotlib.

---
<pre>
multiagent-exploration/
│
├── main.py              -> Entry point: runs the simulation
├── environment.py       -> Environment and map logic
├── agent.py             -> Agent logic and behavior
├── dstar.py             -> D* Lite algorithm implementation
├── map_graph.py         -> Graph abstraction of the environment
├── obstacle.py          -> Obstacle representation
├── utils.py             -> Utilities and helper functions
├── requirements.txt     -> Project dependencies
├── README.md            -> This file
├── todo                 -> Notes or development tasks
</pre>

---

🧠 How It Works
----------------

The MAS simulator uses a decentralized strategy, where each agent explores part of the environment autonomously,
but implicitly coordinates with others via spatial partitioning.

Simulation cycle:

1. Environment Initialization:
   - A probabilistic grid (e.g., 20x20) is created to represent an unknown environment.
   - Obstacles are randomly placed.
   - Agents are distributed in non-overlapping starting positions.

2. Voronoi Partitioning:
   - A Voronoi diagram is computed to divide the environment into regions, one per agent.
   - Each cell is assigned to the nearest agent (Euclidean distance).

3. Frontier Detection:
   - Each agent identifies *frontier cells* within its Voronoi region.
   - Frontier cells are adjacent to unknown cells (probability = 0.5).

4. Goal Selection:
   - The agent selects the frontier cell with the highest entropy (or other heuristic).

5. Path Planning:
   - The agent uses the A* algorithm to compute a path to the selected goal.
   - It follows the path step by step, avoiding obstacles and updating the map.

6. Map Update:
   - At every step, the agent observes its surroundings within a fixed vision range.
   - It updates the occupancy probability of nearby cells.
   - If the probability exceeds a threshold, a cell is considered occupied; otherwise, free.

7. Visualization:
   - The full state of the map, Voronoi regions, frontiers, obstacles, and agent positions
     is visualized in real time using Matplotlib.

8. Repeat:
   - The loop continues for a fixed number of steps or until no unexplored frontier remains.

Each agent is independent but operates on a shared probabilistic map, allowing for efficient distributed exploration.

---

## 🛠️ Dependencies

To run this project, make sure you have Python 3.x installed.

### Install Dependencies

To install all required packages, run:

```bash
pip install -r requirements.txt
