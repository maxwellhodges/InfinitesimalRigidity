import pyximport; pyximport.install()
import cy_testing
import numpy as np


def edge_list_to_adjacency_list(N, edge_list):
    adjacency_list = [[] for node in range(N)]
    for edge in edge_list:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    return adjacency_list


def to_degree_array_and_neighbour_array(adjacency_list):
    degree_array = np.array([len(neighs) for neighs in adjacency_list])
    neighbour_array = np.array([j for i in adjacency_list for j in i])

    return degree_array.astype(np.intc), neighbour_array.astype(np.intc)


N = 8

edge_list = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7)]
adjacency_list = edge_list_to_adjacency_list(N, edge_list)

current_nodes = np.array(range(N)).astype(np.intc)[4:]
adj_list = adjacency_list[4:]
current_adj_list = [list(set(neigh).intersection(set(current_nodes)))
                              for i, neigh in enumerate(adjacency_list) if i in current_nodes]

degs, neighs = to_degree_array_and_neighbour_array(current_adj_list)

triangles = []
for i in range(50):
    triangles.append(cy_testing.find_triangle(degs, neighs, current_nodes))