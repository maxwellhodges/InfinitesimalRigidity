import pyximport; pyximport.install()
import numpy as np
from infrig_helpers import find_triangle


# TODO code won't currently work properly if node is part of multiple clusters
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


def generate_rigidity_matrix(coords, edge_list, N, node_mapping):
    R = np.zeros((len(edge_list), 2 * N))
    for i, edge in enumerate(edge_list):
        node1, node2 = node_mapping[edge[0]], node_mapping[edge[1]]
        R[i, [2 * node1, 2 * node1 + 1]] = coords[node1] - coords[node2]
        R[i, [2 * node2, 2 * node2 + 1]] = coords[node2] - coords[node1]

    return R


def generate_inf_motions(rigidity_matrix, N_current):
    # there are two cases here: edges > 2*nodes (thin) or edges < 2*nodes (fat) TODO look into sparse SVD here
    u, s, vt = np.linalg.svd(rigidity_matrix)
    singular_vals, = np.where(s < 10e-15)
    if rigidity_matrix.shape[0] > rigidity_matrix.shape[1]:
        inf_motions = vt[singular_vals, :]
        # scale the inf motions by the size of the network for numerical stability
        inf_motions = N_current * inf_motions
    else:
        inf_motions = N_current * np.vstack((vt[singular_vals, :], vt[rigidity_matrix.shape[0]:]))

    return inf_motions

# this part of the code will be replaced by function that generates lattice
# number of nodes
# N = 8
# coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]])
# edge_list = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7)]


def cluster_decomp(coords, edge_list):

    N = coords.shape[0]

    adjacency_list = edge_list_to_adjacency_list(N, edge_list)

    node_list = list(range(N))

    # prune any nodes that have fewer than 2 neighbours as cannot be part of rigid cluster
    floppy_nodes = [node for node, neighs in enumerate(adjacency_list) if len(neighs) < 2]
    # current_node_list will keep track of total number of unassigned nodes.  Algorithms ends when it is empty.
    current_node_list = [node for node in node_list if node not in floppy_nodes]
    # also want a list of those nodes with 4 or more edges as these could be involved in multiple clusters
    high_degree_nodes = [node for node, neighs in enumerate(adjacency_list) if len(neighs) > 3]

    current_edge_list = [pair for pair in edge_list if all(node in current_node_list for node in pair)]
    # need a mapping back to the actual node ids
    N_current = len(current_node_list)
    node_mapping = dict(zip(current_node_list, range(N_current)))

    current_coords = np.array([coord for i, coord in enumerate(coords) if i in current_node_list])
    # only want remaining nodes after pruning
    current_adjacency_list = [neigh for i, neigh in enumerate(adjacency_list) if i in current_node_list]

    # create rigidity matrix for structure  TODO need to add logic to reindex after removing floppy nodes
    R = generate_rigidity_matrix(current_coords, current_edge_list, N_current, node_mapping)
    # get the infinitesimal motions
    inf_motions = generate_inf_motions(R, N_current)
    # have another reference to allow successive slicing
    inf_motions_current = inf_motions
    """ The following section will go inside a while loop.  Need to consider that indices need to be matched back to 
    node ids after each iteration as nodes are removed from the node_list """
    cluster_dict = {}
    cluster_index = 0
    triangles = []
    while current_node_list:
        # the current_node_list is the way to map back from list indices to the node value

        # need this format as input into find_triangle function
        current_deg_array, current_neigh_array = to_degree_array_and_neighbour_array(current_adjacency_list)
        # TODO it's now possible to get stuck in an infinite loop by keeping the high degree nodes
        triangle_nodes = find_triangle(current_deg_array, current_neigh_array,
                                       np.array(current_node_list).astype(np.intc))
        if triangle_nodes == -1:
            floppy_nodes.extend(current_node_list)
            break

        # convert triangle_nodes to current indices
        triangle_current_indices = [current_node_list.index(item) for item in triangle_nodes]

        # get three rigid motions from inf motions for the triangle
        rigid_triangle_coords = current_coords[triangle_current_indices, :]
        triangle_centre_of_mass = np.sum(rigid_triangle_coords, axis=0)/3
        distance_from_com = current_coords - triangle_centre_of_mass
        node_rotations = np.cross(distance_from_com, np.array([0, 0, 1]))[:, 0:2]  # note shape of node_rotations is (N, 2)

        # normalise by triangle nodes vector length
        triangle_rotations = node_rotations[np.array(triangle_current_indices)]
        normalisation_factor = np.linalg.norm(triangle_rotations.flatten())

        # rigid motions for all the nodes that are left
        node_rotations_norm = node_rotations.flatten()/normalisation_factor  # flatten node_rotations out here
        node_translations_x_norm = np.tile([1/np.sqrt(3), 0], N_current)
        node_translations_y_norm = np.tile([0, 1/np.sqrt(3)], N_current)

        # rigid motions for triangle
        triangle_rotations_norm = triangle_rotations.flatten()/normalisation_factor
        triangle_translation_x_norm = np.tile([1/np.sqrt(3), 0], 3)
        triangle_translation_y_norm = np.tile([0, 1/np.sqrt(3)], 3)

        # get indices as we have now flattened the motion vectors
        triangle_flattened_indices = [j for i in triangle_current_indices for j in (2*i, 2*i + 1)]
        inf_triangle_motions = inf_motions[:, triangle_flattened_indices]

        # get the translation and rotation components for the nodes for each of the infinitesimal motions
        x_components = inf_triangle_motions.dot(triangle_translation_x_norm)  # TODO can vectorise this part too
        y_components = inf_triangle_motions.dot(triangle_translation_y_norm)
        rot_components = inf_triangle_motions.dot(triangle_rotations_norm)

        # test = inf_triangle_motions - ((x_components.reshape(5, 1) * triangle_translation_x_norm) +
        #                                (y_components.reshape(5, 1) * triangle_translation_y_norm) +
        #                                (rot_components.reshape(5, 1) * triangle_rotations_norm))

        final_differences = inf_motions_current - ((x_components.reshape(inf_motions_current.shape[0], 1) * node_translations_x_norm) +
                                           (y_components.reshape(inf_motions_current.shape[0], 1) * node_translations_y_norm) +
                                           (rot_components.reshape(inf_motions_current.shape[0], 1) * node_rotations_norm))

        temp = final_differences.reshape(inf_motions_current.shape[0], *current_coords.shape)
        absolute_node_distances = np.linalg.norm(temp, axis=2)

        rigid_nodes_boolean = np.min(absolute_node_distances < 1e-6, axis=0)
        rigid_indices = np.where(rigid_nodes_boolean == 1)[0]
        rigid_nodes = np.array(current_node_list)[rigid_indices]

        cluster_dict[cluster_index] = rigid_nodes
        cluster_index += 1

        # remove those nodes found to be rigid  TODO logic isn't quite working
        current_node_list = [node for node in current_node_list if node not in rigid_nodes
                                                                or node in high_degree_nodes]


        N_current = len(current_node_list)
        current_adjacency_list = [list(set(neigh).intersection(set(current_node_list)))
                                  for i, neigh in enumerate(adjacency_list) if i in current_node_list]

        # TODO need to introduce list that retains all nodes with edges >= 4 as these could be in >1 cluster
        # TODO additional, code currently breaks when this is not done as some nodes end up with no neighbours

        current_coords = np.array([coord for i, coord in enumerate(coords) if i in current_node_list])

        # remove motions corresponding to rigid nodes found in current loop
        # TODO this logic now needs to change to reflect keeping the high degree nodes in each loop above
        current_node_indices = [j for i in current_node_list for j in (2 * node_mapping[i], 2 * node_mapping[i] + 1)]
        inf_motions_current = inf_motions[:, current_node_indices]

    return cluster_dict  # TODO also return floppy nodes
