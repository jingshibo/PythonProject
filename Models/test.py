## dynamic time warping
def compute_euclidean_distance_matrix(x, y) -> np.array:
    """Calculate distance matrix
    This method calcualtes the pairwise Euclidean distance between two sequences.
    The sequences can have different lengths.
    """
    dist = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            dist[i,j] = (x[j]-y[i])**2
    return dist


def compute_accumulated_cost_matrix(x, y) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    distances = compute_euclidean_distance_matrix(x, y)

    # Initialization
    cost = np.zeros((len(y), len(x)))
    cost[0, 0] = distances[0, 0]

    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i - 1, 0]

    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j - 1]

        # Accumulated warp path cost
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(cost[i - 1, j],  # insertion
                cost[i, j - 1],  # deletion
                cost[i - 1, j - 1]  # match
            ) + distances[i, j]

    return cost

##
signal_1 = emg_1_mean_events['emg_SSLW_data']
signal_2 = emg_1_mean_channels['emg_SSSA_data'][4]
dtw_distance, warp_path = fastdtw(signal_1, signal_2, dist=euclidean)


cost_matrix = compute_accumulated_cost_matrix(signal_1, signal_2)
# compute the accumulated cost matrix and then visualize the path on a grid.
mpl.rcParams['figure.dpi'] = 20
savefig_options = dict(format="png", dpi=20, bbox_inches="tight")
#
# fig, ax = plt.subplots(figsize=(12, 8))
# ax = sbn.heatmap(cost_matrix, annot=True, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)
# ax.invert_yaxis()
#
# # Get the warp path in x and y directions
# path_x = [p[0] for p in warp_path]
# path_y = [p[1] for p in warp_path]
#
# # Align the path from the center of each cell
# path_xx = [x+0.5 for x in path_x]
# path_yy = [y+0.5 for y in path_y]
#
# ax.plot(path_xx, path_yy, color='blue', linewidth=3, alpha=0.2)
#
# fig.savefig("ex1_heatmap.png", **savefig_options)

# plot the two sequences and connect the mapping points
fig, ax = plt.subplots(figsize=(140, 100))

# Remove the border and axes ticks
fig.patch.set_visible(False)
ax.axis('off')

for [map_x, map_y] in warp_path:
    ax.plot([map_x, map_y], [signal_1[map_x], signal_2[map_y]], linewidth=4)

ax.plot(signal_1, '-ro', label='x', linewidth=4, markersize=20, markerfacecolor='lightcoral', markeredgecolor='lightcoral')
ax.plot(signal_2, '-bo', label='y', linewidth=4, markersize=20, markerfacecolor='skyblue', markeredgecolor='skyblue')
ax.set_title("DTW Distance", fontsize=28, fontweight="bold")
plt.legend()
fig.savefig("ex1_dtw_distance.png", **savefig_options)


##
from dtw import *
signal_1 = emg_1_mean_events['emg_SSLW_data']
signal_1 = emg_1_mean_channels['emg_SASA_data'][0]
signal_2 = emg_1_mean_channels['emg_SASA_data'][4]

alignment = dtw(signal_1, signal_2, keep_internals=True)

# Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")

# Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(signal_1, signal_2, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="twoway",offset=-2)

# See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
rabinerJuangStepPattern(6,"c").plot()
