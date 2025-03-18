import heapq
from utils import stateNameToCoords


def topKey(queue):
    queue.sort()
    # print(queue)
    if len(queue) > 0:
        return queue[0][:2]
    else:
        # print('empty queue!')
        return (float('inf'), float('inf'))


def heuristic_from_s(graph, id, s):
    x_distance = abs(int(id.split('x')[1][0]) - int(s.split('x')[1][0]))
    y_distance = abs(int(id.split('y')[1][0]) - int(s.split('y')[1][0]))
    return max(x_distance, y_distance)


def calculateKey(graph, id, s_current, k_m, agent_id):
    return (min(graph.graph[agent_id][id].g, graph.graph[agent_id][id].rhs) + heuristic_from_s(graph, id, s_current) + k_m, min(graph.graph[agent_id][id].g, graph.graph[agent_id][id].rhs))


def updateVertex(graph, queue, id, s_current, k_m, agent_id):
    s_goal = graph.goals.get(agent_id)
    if id != s_goal:
        min_rhs = float('inf')
        for i in graph.graph[agent_id][id].children:
            if i not in graph.graph[agent_id]:
                print(f"⚠️ ERROR: Node {i} is missing from agent {agent_id}'s graph!")
                print(f"🔍 Existing nodes: {list(graph.graph[agent_id].keys())}")
                print(f"🔄 Current state being processed: {s_current}")
            min_rhs = min(
                min_rhs, graph.graph[agent_id][i].g + graph.graph[agent_id][id].children[i])
        graph.graph[agent_id][id].rhs = min_rhs
    id_in_queue = [item for item in queue if id in item]
    if id_in_queue != []:
        if len(id_in_queue) != 1:
            raise ValueError('more than one ' + id + ' in the queue!')
        queue.remove(id_in_queue[0])
    if graph.graph[agent_id][id].rhs != graph.graph[agent_id][id].g:
        heapq.heappush(queue, calculateKey(graph, id, s_current, k_m, agent_id) + (id,))



def computeShortestPath(graph, queue, s_start, k_m, agent_id):
    print(f"🛠️ Recomputing shortest path from {s_start} | Queue size: {len(queue)}")
    while (graph.graph[agent_id][s_start].rhs != graph.graph[agent_id][s_start].g) or (topKey(queue) < calculateKey(graph, s_start, s_start, k_m, agent_id)):
        if not queue:
            print("Queue is empty, cannot update path!")
            return

        k_old = topKey(queue)
        u = heapq.heappop(queue)[2]

        if k_old < calculateKey(graph, u, s_start, k_m, agent_id):
            heapq.heappush(queue, calculateKey(graph, u, s_start, k_m, agent_id) + (u,))
        elif graph.graph[agent_id][u].g > graph.graph[agent_id][u].rhs:
            graph.graph[agent_id][u].g = graph.graph[agent_id][u].rhs
            for i in graph.graph[agent_id][u].parents:
                updateVertex(graph, queue, i, s_start, k_m, agent_id)
        else:
            graph.graph[agent_id][u].g = float('inf')
            updateVertex(graph, queue, u, s_start, k_m, agent_id)
            for i in graph.graph[agent_id][u].parents:
                updateVertex(graph, queue, i, s_start, k_m, agent_id)
    print(f"✅ Shortest path updated | New queue size: {len(queue)}")


def nextInShortestPath(graph, s_current, agent_id):
    min_rhs = float('inf')
    s_next = None

    if graph.graph[agent_id][s_current].rhs == float('inf'):
        print(f'❌ ERROR: Agent at {s_current} is stuck! No valid path.')
    else:
        for i in graph.graph[agent_id][s_current].children:
            child_cost = graph.graph[agent_id][i].g + graph.graph[agent_id][s_current].children[i]

            # 🔍 Debug print
            print(f"🔎 Checking child {i} | Cost: {child_cost} | Edge cost: {graph.graph[agent_id][s_current].children[i]}")

            if (child_cost) < min_rhs:
                min_rhs = child_cost
                s_next = i

        if s_next:
            print(f"✅ Next move selected: {s_next}")
            return s_next
        else:
            raise ValueError('❌ ERROR: Could not find child for transition!')



def scanForObstacles(graph, queue, s_current, scan_range, k_m, agent_id):
    states_to_update = {} #salva le celle da vedere con la loro probabilità di occupazione
    range_checked = 0
    if scan_range >= 1:# controlla i vicini diretti
        for neighbor in graph.graph[agent_id][s_current].children:
            neighbor_coords = stateNameToCoords(neighbor)
            states_to_update[neighbor] = graph.grid[neighbor_coords[1]
                                                     ][neighbor_coords[0]]
        range_checked = 1
    # print(states_to_update)

    while range_checked < scan_range:
        new_set = {} #salva la lista aggiornata di celle da vedere
        for state in states_to_update:
            new_set[state] = states_to_update[state]
            for neighbor in graph.graph[agent_id][state].children:
                if neighbor not in new_set:
                    neighbor_coords = stateNameToCoords(neighbor)
                    new_set[neighbor] = graph.grid[neighbor_coords[1]
                                                    ][neighbor_coords[0]]
        range_checked += 1
        states_to_update = new_set

    new_obstacle = False
    for state in states_to_update:#controlla tutte le celle viste
        if states_to_update[state] > 0.7:  # found cell with obstacle
            print(f"🔴 Obstacle detected at {state} | Current position: {s_current}")
            for neighbor in graph.graph[agent_id][state].children:
                # first time to observe this obstacle where one wasn't before
                if(graph.graph[agent_id][state].children[neighbor] != float('inf')):
                    print(f"⚠️ Blocking path between {state} and {neighbor}")
                    neighbor_coords = stateNameToCoords(state)
                    #graph.cells[neighbor_coords[1]][neighbor_coords[0]] = -2
                    graph.graph[agent_id][neighbor].children[state] = float('inf')
                    graph.graph[agent_id][state].children[neighbor] = float('inf')
                    updateVertex(graph, queue, state, s_current, k_m, agent_id)
                    new_obstacle = True
        # elif states_to_update[state] == 0: #cell without obstacle
            # for neighbor in graph.graph[state].children:
                # if(graph.graph[state].children[neighbor] != float('inf')):

    # print(graph)
    return new_obstacle


def moveAndRescan(graph, queue, s_current, scan_range, k_m, agent_id):
    if s_current == graph.goals.get(agent_id):
        return 'goal', k_m

    # 1️⃣ Scan first and update the map
    results = scanForObstacles(graph, queue, s_current, scan_range, k_m, agent_id)

    # 2️⃣ Recalculate path if needed
    if results:  # If a new obstacle was detected
        computeShortestPath(graph, queue, s_current, k_m, agent_id)
        print('Obstacle detected, replanning path... , new queue ', queue)

    # 3️⃣ Now select the next move
    s_last = s_current
    print(f"🚶 Agent at {s_current} | Checking next move...")
    s_new = nextInShortestPath(graph, s_current, agent_id)
    print(f"🔍 Next planned move: {s_new}")

    # 4️⃣ Avoid moving into newly discovered obstacles
    new_coords = stateNameToCoords(s_new)
    if graph.grid[new_coords[1]][new_coords[0]] > 0.5:
        print(f"❌ WARNING: {s_new} is an obstacle! Staying at {s_current}")
        s_new = s_current  # Stay in place and wait for replanning

    # 5️⃣ Update key modifier for D* Lite
    k_m += heuristic_from_s(graph, s_last, s_new)

    # 6️⃣ Compute shortest path as usual (only if no obstacle was detected earlier)
    computeShortestPath(graph, queue, s_current, k_m, agent_id)
    #print('Moving from', s_current, 'to', s_new, 'with k_m =', k_m, 'new queue', queue)

    return s_new, k_m



def initDStarLite(graph, queue, s_start, s_goal, k_m, agent_id):
    graph.graph[agent_id][s_goal].rhs = 0
    heapq.heappush(queue, calculateKey(
        graph, s_goal, s_start, k_m, agent_id) + (s_goal,))
    computeShortestPath(graph, queue, s_start, k_m, agent_id)

    return (graph, queue, k_m)