from libc.math cimport sqrt

from cymem.cymem cimport Pool
cimport cython

import numpy as np


ctypedef struct node_info:
    int degree
    int* neighs

cdef Pool mem = Pool()

# @cython.boundscheck(False)
# @cython.wraparound(False)
def find_triangle(int[:] degrees, int[:] neighbours): # TODO allow different dtypes in code?

    nnodes = len(degrees)

    cdef node_info* nodes = <node_info*>mem.alloc(nnodes, sizeof(node_info))
    cdef int i, j, k, l

    cdef int index = 0
    for i in range(nnodes):
        nodes[i].degree = degrees[i]
        nodes[i].neighs = &neighbours[index]
        index += degrees[i]

    # now need to do a brute force search to find triangle
    # np.random.seed(42)
    node_indices = np.random.permutation(range(nnodes))

    cdef:
        int node
        int neigh_node
        int neighneigh_node
        int neighneighneigh_node
    for node in node_indices:
        for j in range(nodes[node].degree):
            neigh_node = nodes[node].neighs[j]
            for k in range(nodes[neigh_node].degree):
                neighneigh_node = nodes[neigh_node].neighs[k]
                for l in range(nodes[neighneigh_node].degree):
                    neighneighneigh_node = nodes[neighneigh_node].neighs[l]
                    if neighneighneigh_node == node:
                        return node, neigh_node, neighneigh_node

    return -1

# TODO if a node doesn't move back in a single inf motion, then it can be removed from the loop
# think i only need to do the SVD once and then can re-use rigid motions for triangles from that
# def find_cluster(double[:, :] inf_motions, double[:, :] triangle_rigid_motions):

# def rotation_vector(int[:] triangle_nodes, double[:,:] triangle_coords, double[:,:] node_coords):
#
#     cdef int N = node_coords.shape[0] # TODO change variable name - confusing
#     # return a 2N-vector here that has has the 2D rotational vector for each node
#     rotations = np.zeros(N, 2, dtype=np.float)
#     cdef double[:, :] rotations_view = rotations # TODO turn this into 2D vector
#
#     cdef int i
#
#     cdef double centre_of_mass[2]
#     centre_of_mass[0] = (triangle_coords[0, 0] + triangle_coords[1, 0] + triangle_coords[2, 0])/3 # TODO use C division
#     centre_of_mass[1] = (triangle_coords[0, 1] + triangle_coords[1, 1] + triangle_coords[2, 1])/3 # TODO use C division
#
#     for i in range(N): # TODO make this a C division
#         rotations[i, 0] = node_coords[i, 1] - centre_of_mass[1]
#         rotations[i, 1] = centre_of_mass[0] - node_coords[i, 0]
#
#
#     cdef int node
#     cdef double temp1, temp2, normalisation_factor
#     normalisation_factor = 0.0
#     triangle_rotations = np.zeros(6)
#     for i in range(3):
#         node = triangle_nodes[i]
#         temp1, temp2 = rotations[2 * node], rotations[2* node + 1]
#         triangle_rotations[2*i], triangle_rotations[2*i + 1] = temp1, temp2
#         normalisation_factor += temp1*temp1 + temp2*temp2
#     normalisation_factor = sqrt(normalisation_factor)
#
#     return triangle_rotations/normalisation_factor, rotations/normalisation_factor  # TODO use C division
#     # return rotations

# # TODO test this with a single inf_motion first
# def find_cluster(int[:] triangle_nodes, double[:] inf_motion, double[:] rotations):
#
#     N = len(inf_motion)
#
#     cdef:
#         double triangle_inf_motions[6]
#         double triangle_rot_motions[6]
#         int i
#         int node
#     for i in range(3):
#         node = triangle_nodes[i]
#         triangle_inf_motions[2*i] = inf_motion[2*node]
#         triangle_inf_motions[2*i + 1] = inf_motion[2*node+1]
#
#         triangle_rot_motions[2*i] = rotations[2*node]
#         triangle_rot_motions[2*i + 1] = rotations[2*node+1]
#
#     # cdef:
#     #     double rigid_x_motions = [1/sqrt(3), 0, 1/sqrt(3), 0, 1/sqrt(3), 0]
#     #     double rigid_y_motions = [0, 1/sqrt(3), 0, 1/sqrt(3), 0, 1/sqrt(3)]
#
#     # get components of triangle rigid motion in each 'direction'
#     cdef double rigid_y_component, rigid_x_component
#     multiple = np.zeros(6)
#     rigid_x_component = (triangle_inf_motions[0] +
#                          triangle_inf_motions[2] +
#                          triangle_inf_motions[4])/3
#     rigid_y_component = (triangle_inf_motions[1] +
#                          triangle_inf_motions[3] +
#                          triangle_inf_motions[5])/3
#     # do it this way for now, try to use dot products so generalises to 3D afterwards
#     for i in range(3):
#         multiple[2*i] = (triangle_inf_motions[2*i] - rigid_x_component)/triangle_rot_motions[2*i]
#         multiple[2*i + 1] = (triangle_inf_motions[2*i + 1] - rigid_y_component)/triangle_rot_motions[2*i + 1]
#
#     # TODO decide what to do about indices here to make it clear
#     # set everything to 1, then change to 0 when atom found to be flexible
#     rigid_atoms = np.ones(N)
#     cdef double x, y
#     x,y = 0.0, 0.0
#     for i in range(N/2):
#         rigid_atoms[2*i] = inf_motion[2*i] - rigid_x_component - (multiple[0] * rotations[2*i])
#         rigid_atoms[2*i + 1] = inf_motion[2*i + 1] - rigid_y_component - (multiple[0] * rotations[2*i+1])
#         # if sqrt(x*x + y*y) > 10e-6:
#         #     rigid_atoms[i] = 0
#
#     return rigid_atoms
