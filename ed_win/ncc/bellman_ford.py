# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 09:30:26 2021

@author: juru
"""
import numpy as np
# Structure to represent a weighted
# edge in graph


class Edge:
    def __init__(self):
        self.src = 0
        self.dest = 0
        self.weight = 0
# Structure to represent a directed
# and weighted graph


class Graph:
    def __init__(self):
        # V. Number of vertices, E.
        # Number of edges
        self.V = 0
        self.E = 0
        # Graph is represented as
        # an array of edges.
        self.edge = []
# Creates a new graph with V vertices
# and E edges


def createGraph(V, E):
    graph = Graph()
    graph.V = V
    graph.E = E
    graph.edge = [Edge() for i in range(graph.E)]
    return graph
# Function runs Bellman-Ford algorithm
# and prints negative cycle(if present)


def neg_cycle_bellman_ford(vertices, graph, src):
    V = graph.V
    E = graph.E
    # %%
    dist = {}
    for i in vertices:
        dist[i] = 100000000000
    # %%
    #dist =[100000000000 for i in range(V)]
    #parent =[-1 for i in range(V)]
    # %%
    parent = {}
    for i in vertices:
        parent[i] = -1
    # %%
    dist[src] = 0
    # Relax all edges |V| - 1 times
    for i in range(1, V):
        for j in range(E):
            u = graph.edge[j].src
            v = graph.edge[j].dest
            weight = graph.edge[j].weight
            if (dist[u] != 100000000000 and dist[u] + weight < dist[v]):
                dist[v] = dist[u] + weight
                parent[v] = u
    # Check for negative-weight cycles
    C = -1
    for i in range(E):
        u = graph.edge[i].src
        v = graph.edge[i].dest
        weight = graph.edge[i].weight
        if (dist[u] != 100000000000 and dist[u] + weight < dist[v]):
            # Store one of the vertex of the negative weight cycle
            C = v
            break
    cycle = []
    if (C != -1):
        for i in range(V):
            C = parent[C]
        # To store the cycle vertex
        v = C
        while (True):
            # cycle.append(v)
            if (v == C and len(cycle) > 1):
                break
            cycle.append(v)
            v = parent[v]
            cycle.append(v)
        # Reverse cycle[]
        cycle.reverse()
    return np.array(cycle)


# Driver Code
if __name__ == '__main__':		# Number of vertices in graph
    #V = 5
    # Number of edges in graph
    #E = 5
    #graph = createGraph(V, E)
    # Given Graph
    #graph.edge[0].src = 0
    #graph.edge[0].dest = 1
    #graph.edge[0].weight = 1

    #graph.edge[1].src = 1
    #graph.edge[1].dest = 2
    #graph.edge[1].weight = 2

    #graph.edge[2].src = 2
    #graph.edge[2].dest = 3
    #graph.edge[2].weight = 3

    #graph.edge[3].src = 3
    #graph.edge[3].dest = 4
    #graph.edge[3].weight = -3

    #graph.edge[4].src = 4
    #graph.edge[4].dest = 1
    #graph.edge[4].weight = -3

    # Function Call
    #cycle=neg_cycle_bellman_ford(np.array([0,1,2,3,4]),graph, 0)
    import time
    t = time.time()
    V = 7
    E = 12 + 6
    graph = createGraph(V, E)
    graph.edge[0].src = -1
    graph.edge[0].dest = 1
    graph.edge[0].weight = 0
    graph.edge[1].src = -1
    graph.edge[1].dest = 2
    graph.edge[1].weight = 0
    graph.edge[2].src = -1
    graph.edge[2].dest = 3
    graph.edge[2].weight = 0
    graph.edge[3].src = -1
    graph.edge[3].dest = 4
    graph.edge[3].weight = 0
    graph.edge[4].src = -1
    graph.edge[4].dest = 5
    graph.edge[4].weight = 0
    graph.edge[5].src = -1
    graph.edge[5].dest = 6
    graph.edge[5].weight = 0
    graph.edge[6].src = 1
    graph.edge[6].dest = 2
    graph.edge[6].weight = 1
    graph.edge[7].src = 2
    graph.edge[7].dest = 1
    graph.edge[7].weight = -1
    graph.edge[8].src = 3
    graph.edge[8].dest = 4
    graph.edge[8].weight = 1
    graph.edge[9].src = 6
    graph.edge[9].dest = 5
    graph.edge[9].weight = 1
    graph.edge[10].src = 5
    graph.edge[10].dest = 6
    graph.edge[10].weight = -1
    graph.edge[11].src = 1
    graph.edge[11].dest = 4
    graph.edge[11].weight = 5
    graph.edge[12].src = 4
    graph.edge[12].dest = 1
    graph.edge[12].weight = -5
    graph.edge[13].src = 4
    graph.edge[13].dest = 5
    graph.edge[13].weight = 4
    graph.edge[14].src = 5
    graph.edge[14].dest = 4
    graph.edge[14].weight = -4
    graph.edge[15].src = 2
    graph.edge[15].dest = 3
    graph.edge[15].weight = 2
    graph.edge[16].src = 3
    graph.edge[16].dest = 2
    graph.edge[16].weight = -2
    graph.edge[17].src = 6
    graph.edge[17].dest = 3
    graph.edge[17].weight = -3

    cycle = neg_cycle_bellman_ford(np.array([-1, 1, 2, 3, 4, 5, 6]), graph, -1)
    tot_time = 1000 * (time.time() - t)
    print('Total time in milisecods', tot_time)
