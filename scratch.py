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
distances[np.triu_indices(num_points)] = 10  # removes repeats and diagonal values
indices = np.where(distances < np.sqrt(2) + 0.0001)
edges_vector = np.vstack([*indices]).T

# def unique(sequence):
#     seen = set()
#     return [x for x in sequence if not (x in seen or seen.add(x))]


allowed_edges = [tuple(pair) for pair in edges_vector.tolist()]

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
subset_of_edges = subset_edges(allowed_edges, 0.3)
plot_lattice(x, y, subset_of_edges)

import infrig
# TODO need to update new_triangle code with test version that seems to work now
out = infrig.cluster_decomp(points, subset_of_edges)

colors=["#008000", "#FF0000", "#0000FF", "#FFBD33"]

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

