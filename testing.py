import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import combinations

# number of nodes
N = 8

# node number is index here
toy_structure_coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]])

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
R_triangle = np.zeros((len(edge_list), 2 * len(toy_structure_coords)))
for i, edge in enumerate(edge_list):
    node1, node2 = edge[0], edge[1]
    node1_coord, node2_coord = toy_structure_coords[node1], toy_structure_coords[node2]
    R_triangle[i, [2 * node1, 2 * node1 + 1]] = (np.array(node1_coord) - np.array(node2_coord))
    R_triangle[i, [2 * node2, 2 * node2 + 1]] = (np.array(node1_coord) - np.array(node2_coord))


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
triangle_nodes = find_triangle(degs.astype(np.int32), neighs.astype(np.int32))

# construct rigidity matrix for triangle to get the rigid motions
# wrap all this into a function
R_triangle = np.zeros((3, 6))
R_triangle[0, 0:2] = toy_structure_coords[triangle_nodes[0]] - toy_structure_coords[triangle_nodes[1]]
R_triangle[0, 2:4] = toy_structure_coords[triangle_nodes[1]] - toy_structure_coords[triangle_nodes[0]]

R_triangle[1, 0:2] = toy_structure_coords[triangle_nodes[0]] - toy_structure_coords[triangle_nodes[2]]
R_triangle[1, 4:] = toy_structure_coords[triangle_nodes[2]] - toy_structure_coords[triangle_nodes[0]]

R_triangle[2, 2:4] = toy_structure_coords[triangle_nodes[1]] - toy_structure_coords[triangle_nodes[2]]
R_triangle[2, 4:] = toy_structure_coords[triangle_nodes[2]] - toy_structure_coords[triangle_nodes[1]]

# individual directions of nullspace are arbitrary, they are combinations of rigid translations and rotations
u_tri, s_tri, vt_tri = np.linalg.svd(R_triangle)
nullspace = vt_tri[3:, :]

# create rigidity matrix for whole structure
R = np.zeros((len(edge_list), 2 * N))
for i, edge in enumerate(edge_list):
    node1, node2 = edge[0], edge[1]
    R[i, [2 * node1, 2 * node1+1]] = toy_structure_coords[node1] - toy_structure_coords[node2]
    R[i, [2 * node2, 2 * node2 + 1]] = toy_structure_coords[node2] - toy_structure_coords[node1]

# get the infinitesimal motions
# there are two cases here: edges > 2*nodes (thin) or edges < 2*nodes (fat) TODO look into sparse SVD here
u, s, vt = np.linalg.svd(R)
singular_vals, = np.where(s < 10e-15)
if R.shape[0] > R.shape[1]:
    inf_motions = vt[singular_vals, :]
    # scale the inf motions by the size of the network for numerical stability
    inf_motions = N * inf_motions
else:
    inf_motions = np.vstack((vt[singular_vals, :], vt[R.shape[0]:]))

# remove those that are floppy (nedges < 3) then find a triangle
# all nodes that move back to their original positions for all inf motions are part
# of same cluster as triangle.  Remove all of these from list and find another triangle etc
node_set = set(range(N))
floppy_nodes, = np.where(degs < 2)
node_set = node_list.difference(floppy_nodes.tolist())

# want a cython function here that accepts inf motions of remaining nodes and returns those
# nodes in a single cluster.  Will be a while loop

rigid_nodes = cy_testing.find_cluster()