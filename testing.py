import matplotlib.pyplot as plt
import numpy as np
import random

# number of nodes
N = 8

# node number is index here
toy_structure_coords = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]]

x = [element[0] for element in toy_structure_coords]
y = [element[1] for element in toy_structure_coords]



edge_list = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)] + \
            [(4, 5), (4, 6), (5, 6), (5, 7), (6, 7)]

# for edge in edge_list:
#     node1, node2 = edge[0], edge[1]
#     coord1, coord2 = toy_structure_coords[node1], toy_structure_coords[node2]
#     plt.plot(np.array([coord1[0], coord2[0]]), np.array([coord1[1], coord2[1]]), color='b')
#
# plt.plot(np.array(x), np.array(y), 'ro')
#
# plt.show()

# create the rigidity matrix for 2D structure
R = np.zeros((len(edge_list), 2 * len(toy_structure_coords)))
for i, edge in enumerate(edge_list):
    node1, node2 = edge[0], edge[1]
    node1_coord, node2_coord = toy_structure_coords[node1], toy_structure_coords[node2]
    R[i, [2 * node1, 2 * node1 + 1]] = (np.array(node1_coord) - np.array(node2_coord))
    R[i, [2 * node2, 2 * node2 + 1]] = (np.array(node1_coord) - np.array(node2_coord))


# create array structures for cython code
def to_degree_array_and_neighbour_array(nnodes, edge_list):

    adjacency_list = [[] for node in range(nnodes)]
    for edge in edge_list:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    degree_array = np.array([len(neighs) for neighs in adjacency_list])
    neighbour_array = np.array([j for i in adjacency_list for j in i])

    return degree_array, neighbour_array


degs, neighs = to_degree_array_and_neighbour_array(N, edge_list)

test_edge_list = [(0, 1), (1, 2), (2, 3)]

degs_test, neigh_test = to_degree_array_and_neighbour_array(4, test_edge_list)

from importlib import reload
import pyximport; pyximport.install(reload_support=True)

from cy_testing import find_triangle
