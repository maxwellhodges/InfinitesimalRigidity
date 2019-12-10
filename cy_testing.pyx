from cymem.cymem cimport Pool
cimport cython

import numpy as np


ctypedef struct node_info:
    int degree
    int* neighs

cdef Pool mem = Pool()

# @cython.boundscheck(False)
# @cython.wraparound(False)
def find_triangle(int[:] degrees, int[:] neighbours):

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

    return "No triangles found"


