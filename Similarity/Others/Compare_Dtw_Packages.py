from dtw import accelerated_dtw
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import matplotlib.pyplot as plt
import datetime
import numpy as np


## dtaidistance package for two sequence comparison
signal_1 = emg_1_mean_channels['emg_SSLW_data'][0]
signal_2 = emg_1_mean_channels['emg_SSSA_data'][4]
distance, paths = dtw.warping_paths_fast(signal_1, signal_2)
print(distance)
print(paths)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(signal_1, signal_2, paths, best_path) # plot cost matrix

# plot the two sequences and connect the mapping points
fig = plt.figure()
ax = plt.axes()
# Remove the border and axe sticks
fig.patch.set_visible(False)
ax.axis('off')
for [map_x, map_y] in best_path:
    ax.plot([map_x, map_y], [signal_1[map_x], signal_2[map_y]], linewidth=4)

ax.plot(signal_1, '-ro', label='x', linewidth=4, markersize=20, markerfacecolor='lightcoral', markeredgecolor='lightcoral')
ax.plot(signal_2, '-bo', label='y', linewidth=4, markersize=20, markerfacecolor='skyblue', markeredgecolor='skyblue')
ax.set_title("DTWDistance", fontsize=28, fontweight="bold")
plt.legend()

##  dtaidistance package for comparison between a Set of Series
now = datetime.datetime.now()
dtw_results = []
signal = emg_1_mean_channels['emg_SASS_data']
dtw_results = dtw.distance_matrix_fast(signal, block=((0, 1), (1, len(signal))), compact=True)
print(datetime.datetime.now() - now)

## dwt package: accelerated_dtw method
signal_1 = emg_1_mean_channels['emg_SSLW_data'][0]
signal_2 = emg_1_mean_channels['emg_SSSA_data'][4]
d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(signal_1, signal_2, dist='euclidean')

# plot cost matrix
plt.figure()
plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlabel('Subject1')
plt.ylabel('Subject2')
plt.title(f'DTW Minimum Path with minimumdistance:{np.round(d, 2)}')
plt.show()

# plot the two sequences and connect the mapping points
fig = plt.figure()
ax = plt.axes()
# Remove the border and axe sticks
fig.patch.set_visible(False)
ax.axis('off')

path_pair = zip(path[0], path[1])
for [map_x, map_y] in path_pair:
    ax.plot([map_x, map_y], [signal_1[map_x], signal_2[map_y]], linewidth=4)

ax.plot(signal_1, '-ro', label='x', linewidth=4, markersize=20, markerfacecolor='lightcoral', markeredgecolor='lightcoral')
ax.plot(signal_2, '-bo', label='y', linewidth=4, markersize=20, markerfacecolor='skyblue', markeredgecolor='skyblue')
ax.set_title("DTWDistance", fontsize=28, fontweight="bold")
plt.legend()
