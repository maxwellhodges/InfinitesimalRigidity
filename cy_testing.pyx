from libc.math cimport sqrt
from libc.stdlib cimport qsort

from cymem.cymem cimport Pool
cimport cython

import numpy as np


ctypedef struct node_info:
    int degree
    int* neighs

cdef Pool mem = Pool()

# @cython.boundscheck(False)
# @cython.wraparound(False)
def find_triangle(int[:] degrees, int[:] neighbours, int[:] current_nodes): # TODO allow different dtypes in code?

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
        int node, neigh_node, neighneigh_node, neighneighneigh_node
        int node_index, neigh_node_index, neighneigh_node_index

    for node_index in node_indices:
        node = current_nodes[node_index]
        for j in range(nodes[node_index].degree):
            neigh_node = nodes[node_index].neighs[j]
            neigh_node_index = index_testing(current_nodes, neigh_node)

            for k in range(nodes[neigh_node_index].degree):
                neighneigh_node = nodes[neigh_node_index].neighs[k]
                neighneigh_node_index = index_testing(current_nodes, neighneigh_node)

                for l in range(nodes[neighneigh_node_index].degree):
                    neighneighneigh_node = nodes[neighneigh_node_index].neighs[l]
                    if neighneighneigh_node == node:
                        return node, neigh_node, \
                               neighneigh_node

    return -1

cdef int index_testing(int[:] current_node_list, int node):

    cdef int i, length, index

    length = len(current_node_list)

    index = 0
    for i in range(length):
        if current_node_list[i] == node:
            break
        index += 1  # TODO need to return something if go over end of array

    return index


def test(int[:] rigid_nodes, int[:] triangle):
    """ Returns 1 if triangle is not in rigid_nodes and 0 otherwise"""
    # rigid_nodes will be presorted
    cdef:
        int nnodes = rigid_nodes.shape[0]
        int i, j

    cdef int* tri_matches = <int*>mem.alloc(3, sizeof(int))
    qsort(&triangle[0], triangle.shape[0], triangle.strides[0], &cmp_func)
    j = 0
    for i in range(nnodes):
        if rigid_nodes[i] == triangle[j]:
            tri_matches[j] = 1
            j += 1

    for i in range(3):
        if tri_matches[i] == 0:
            return 1

    return 0



def new_triangle(int[:] triangle, int[:,:] triangle_list):
    """Returns true if triangle has not already been identified"""
    cdef:
        int rows = triangle_list.shape[0]
        int i, j

    # sort the triangle nodes
    qsort(&triangle[0], 3, sizeof(int), &cmp_func)

    # make element of matches 1 if rows doesn't match
    cdef int* matches = <int*>mem.alloc(rows, sizeof(int))
    for i in range(rows):
        qsort(&triangle_list[i, 0], triangle_list[i].shape[0], triangle_list[i].strides[0], &cmp_func)
        for j in range(3):
            if triangle_list[i, j] != triangle[j]:
                matches[i] = 1

    for i in range(rows):
        if matches[i] == 0:
            return 0

    return 1


cdef int cmp_func(const void* a, const void* b) nogil:
    cdef int a_v = (<int*>a)[0]
    cdef int b_v = (<int*>b)[0]
    if a_v < b_v:
        return -1
    elif a_v == b_v:
        return 0
    else:
        return 1


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



