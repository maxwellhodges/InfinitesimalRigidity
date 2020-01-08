import pyximport; pyximport.install()
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


# create array structures for cython code TODO change this to return adjacency list
def to_degree_array_and_neighbour_array(nnodes, edge_list):

    adjacency_list = [[] for node in range(nnodes)]
    for edge in edge_list:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    degree_array = np.array([len(neighs) for neighs in adjacency_list])
    neighbour_array = np.array([j for i in adjacency_list for j in i])

    return degree_array, neighbour_array


degs, neighs = to_degree_array_and_neighbour_array(N, edge_list)

# test_edge_list = [(0, 1), (1, 2), (2, 3)]
#
# degs_test, neigh_test = to_degree_array_and_neighbour_array(4, test_edge_list)


from importlib import reload
import pyximport; pyximport.install(reload_support=True)

import cy_testing
triangle_nodes = cy_testing.find_triangle(degs.astype(np.int32), neighs.astype(np.int32))


# create rigidity matrix for whole structure
R1 = np.zeros((len(edge_list), 2 * N))
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
floppy_nodes, = np.where(degs < 2)  # TODO can just become python expression if using adjacency list
node_set = node_set.difference(floppy_nodes.tolist())

# original adjacency list - need to refactor all this to make adjacency list the main structure
adjacency_list = [[] for node in range(N)]
for edge in edge_list:
    adjacency_list[edge[0]].append(edge[1])
    adjacency_list[edge[1]].append(edge[0])

# only want remaining nodes after pruning
current_adjacency_list = [neigh for i, neigh in enumerate(adjacency_list) if i in node_set]
current_deg_list = [deg for i, deg in enumerate(degs) if i in node_set]

current_degree_array = np.array([len(neighs) for neighs in adjacency_list])
current_neighbour_array = np.array([j for i in adjacency_list for j in i])

# want a cython function here that accepts inf motions of remaining nodes and returns those
# nodes in a single cluster.  Will be a while loop
# note, this will return the indices of the *remaining* nodes so need to convert back
triangle_nodes = cy_testing.find_triangle(current_degree_array.astype(np.int32),
                                          current_neighbour_array.astype(np.int32))

# get three rigid motions from inf motions for the triangle
rigid_triangle_coords = toy_structure_coords[triangle_nodes, :]

# try this just using numpy
triangle_centre_of_mass = np.sum(rigid_triangle_coords, axis=0)/3
distance_from_com = toy_structure_coords - triangle_centre_of_mass
node_rotations = np.cross(distance_from_com, np.array([0, 0, 1]))[:, 0:2]

# need to normalise by triangle nodes
triangle_rotations = node_rotations[np.array(triangle_nodes)]
normalisation_factor = np.linalg.norm(triangle_rotations.flatten())

node_rotations_norm = node_rotations/normalisation_factor
node_translations_x_norm = np.tile([1/np.sqrt(3), 0], N)
node_translations_y_norm = np.tile([0, 1/np.sqrt(3)], N)

triangle_rotations_norm = triangle_rotations/normalisation_factor
triangle_translation_x_norm = np.tile([1/np.sqrt(3), 0], 3)
triangle_translation_y_norm = np.tile([0, 1/np.sqrt(3)], 3)

"""This is the section we want to parallelise - should be able to use numpy broadcasting"""
# for a single inf_motion, find the component in each rigid direction for the triangle
# test_inf = inf_motions[0]
# test_inf_triangle = test_inf.reshape(*toy_structure_coords.shape)[np.array(triangle_nodes)].flatten()

triangle_node_indices = [j for i in triangle_nodes for j in (2*i, 2*i + 1)]
inf_triangle_motions = inf_motions[:, triangle_node_indices]

x_components = inf_triangle_motions.dot(triangle_translation_x_norm)  # TODO can vectorise this part too
y_components = inf_triangle_motions.dot(triangle_translation_y_norm)
rot_components = inf_triangle_motions.dot(triangle_rotations_norm.flatten())

# make sure we get all zeros here
test = inf_triangle_motions - ((x_components.reshape(5, 1) * triangle_translation_x_norm) +
                               (y_components.reshape(5, 1) * triangle_translation_y_norm) +
                               (rot_components.reshape(5, 1) * triangle_rotations_norm.flatten()))

final_differences = inf_motions - ((x_components.reshape(5, 1) * node_translations_x_norm) +
                                (y_components.reshape(5, 1) * node_translations_y_norm) +
                                (rot_components.reshape(5, 1) * node_rotations_norm.flatten()))

temp = final_differences.reshape(inf_motions.shape[0], *toy_structure_coords.shape)
absolute_node_distances = np.linalg.norm(temp, axis=2)
floppy_nodes_boolean = np.max(absolute_node_distances > 1e-5, axis=0)


degs = np.array([3,3,3,2]).astype(np.intc)
neigh = np.array([3,5,6,4,6,7,4,5,7,5,6]).astype(np.intc)


import pyximport; pyximport.install()
import numpy as np

triangles = np.array([[3,2,1],[4,6,5],[7,9,8]]).astype(np.intc)
triangle = np.array([8,7,9]).astype((np.intc))
from cy_testing import new_triangle
triangle2 = np.array([10,12,11]).astype((np.intc))