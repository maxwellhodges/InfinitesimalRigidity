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


def generate_rigidity_matrix(edge_list, N):
    R = np.zeros((len(edge_list), 2 * N))
    for i, edge in enumerate(edge_list):
        node1, node2 = edge[0], edge[1]
        R[i, [2 * node1, 2 * node1 + 1]] = toy_structure_coords[node1] - toy_structure_coords[node2]
        R[i, [2 * node2, 2 * node2 + 1]] = toy_structure_coords[node2] - toy_structure_coords[node1]

    return R


def generate_inf_motions(rigidity_matrix):
    # there are two cases here: edges > 2*nodes (thin) or edges < 2*nodes (fat) TODO look into sparse SVD here
    u, s, vt = np.linalg.svd(rigidity_matrix)
    singular_vals, = np.where(s < 10e-15)
    if R.shape[0] > R.shape[1]:
        inf_motions = vt[singular_vals, :]
        # scale the inf motions by the size of the network for numerical stability
        inf_motions = N * inf_motions
    else:
        inf_motions = np.vstack((vt[singular_vals, :], vt[R.shape[0]:]))

    return inf_motions

# this part of the code will be replaced by function that generates lattice
# number of nodes
N = 8
toy_structure_coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]])
edge_list = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7)]

adjacency_list = edge_list_to_adjacency_list(N, edge_list)

# node_set will keep track of total number of unassigned nodes.  Algorithms ends when it is empty.
node_set = set(range(N))

# prune any nodes that have fewer than 2 neighbours as cannot be part of rigid cluster
floppy_nodes = set(node for node, neighs in enumerate(adjacency_list) if len(neighs) < 2)
node_set = node_set.difference(floppy_nodes)


# create rigidity matrix for whole structure TODO this should only be non-dangling nodes for efficiency
R = generate_rigidity_matrix(edge_list, N)
# get the infinitesimal motions
inf_motions = generate_inf_motions(R)

# only want remaining nodes after pruning
current_adjacency_list = [neigh for i, neigh in enumerate(adjacency_list) if i in node_set]

# need this format as input into find_triangle function
deg_array, neigh_array = to_degree_array_and_neighbour_array(current_adjacency_list)





