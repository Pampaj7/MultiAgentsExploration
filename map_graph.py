class Node:
    def __init__(self, id):
        self.id = id
        # dictionary of parent node ID's
        # key = id of parent
        # value = (edge cost,)
        self.parents = {}
        # dictionary of children node ID's
        # key = id of child
        # value = (edge cost,)
        self.children = {}
        # g approximation
        self.g = float('inf')
        # rhs value
        self.rhs = float('inf')

    def __str__(self):
        return 'Node: ' + self.id + ' g: ' + str(self.g) + ' rhs: ' + str(self.rhs)

    def __repr__(self):
        return self.__str__()

    def update_parents(self, parents):
        self.parents = parents


class Graph:
    def __init__(self):
        self.graph = {}
        self.start = {}  # Dictionary to store start nodes for each agent
        self.goals = {}   # Dictionary to store goal nodes for each agent

    def __str__(self):
        msg = 'Graph:'
        for i in self.graph:
            msg += '\n  node: ' + i + ' g: ' + \
                str(self.graph[i].g) + ' rhs: ' + str(self.graph[i].rhs)
        return msg

    def __repr__(self):
        return self.__str__()

    def setStart(self, id, agent_id):
        if id in self.graph:  # Check if the node exists in the graph
            self.start[agent_id] = id
        else:
            raise ValueError(f'Start ID {id} not found in graph.')

    def setGoal(self, id, agent_id):
        if id in self.graph:  # Check if the node exists in the graph
            self.goals[agent_id] = id
        else:
            raise ValueError(f'Goal ID {id} not found in graph.')
        
    def addNodeToGraph(self, id, neighbors, edge=1):
        node = Node(id)
        for i in neighbors:
            # print(i)
            node.parents[i] = edge
            node.children[i] = edge
        self.graph[id] = node
        
