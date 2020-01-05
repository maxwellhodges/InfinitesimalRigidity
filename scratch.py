# import pyximport; pyximport.install()
# import cy_testing
# import numpy as np
#
#
# def edge_list_to_adjacency_list(N, edge_list):
#     adjacency_list = [[] for node in range(N)]
#     for edge in edge_list:
#         adjacency_list[edge[0]].append(edge[1])
#         adjacency_list[edge[1]].append(edge[0])
#
#     return adjacency_list
#
#
# def to_degree_array_and_neighbour_array(adjacency_list):
#     degree_array = np.array([len(neighs) for neighs in adjacency_list])
#     neighbour_array = np.array([j for i in adjacency_list for j in i])
#
#     return degree_array.astype(np.intc), neighbour_array.astype(np.intc)
#
#
# N = 8
#
# edge_list = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7)]
# adjacency_list = edge_list_to_adjacency_list(N, edge_list)
#
# current_nodes = np.array(range(N)).astype(np.intc)[4:]
# adj_list = adjacency_list[4:]
# current_adj_list = [list(set(neigh).intersection(set(current_nodes)))
#                               for i, neigh in enumerate(adjacency_list) if i in current_nodes]
#
# degs, neighs = to_degree_array_and_neighbour_array(current_adj_list)
#
# triangles = []
# for i in range(50):
#     triangles.append(cy_testing.find_triangle(degs, neighs, current_nodes))


import itertools
import numpy as np
import matplotlib.pyplot as plt

points = np.array(list(itertools.product(range(6), range(6))))
num_points = points.shape[0]

x = points[:, 0]
y = points[:, 1]

# get total edge list
import scipy.spatial.distance as ssd
distances = ssd.cdist(points, points)
indices = np.where(distances < np.sqrt(2) + 0.0001)
edges_repeats = np.vstack([*indices]).T

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


allowed_edges = unique(tuple(pair) for pair in edges_repeats.tolist())

# plot figure with edges present
def plot_lattice(x, y, allowed_edges=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if allowed_edges is not None:
        for edge in allowed_edges:
            ax.plot(x[list(edge)], y[list(edge)], 'b-')
    ax.plot(x, y, 'ro')
    plt.show()

# take random subsets of the allowed edges
def subset_edges(edges, fraction_edges=0.5):
    num_allowed_edges = len(edges)
    np.random.seed(42)
    num_subset_edges = int(fraction_edges * num_allowed_edges)
    subset_indices = np.random.permutation(num_allowed_edges)[:num_subset_edges]
    subset_edges = [allowed_edges[i] for i in subset_indices]
    return subset_edges


# plot figure with subset edges present
subset_of_edges = subset_edges(allowed_edges, 0.2)
plot_lattice(x, y, subset_of_edges)

import infrig
out = infrig.cluster_decomp(points, subset_of_edges)

colors=["#008000", "#FF0000", "#0000FF"]

col = [None] * num_points
for key in out.keys():
    for node in out[key]:
        col[node] = colors[key]

col = ["#808080" if v is None else v for v in col]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for edge in subset_of_edges:
    ax.plot(x[list(edge)], y[list(edge)], c='grey', alpha=0.4)
for i in range(num_points):
    ax.scatter(x[i], y[i], color=col[i], s=100)
plt.show()
# plt.savefig("edge_prob_03.png")

