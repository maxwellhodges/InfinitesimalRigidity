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