## model accuracy results
raw_cnn_accuracy = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
raw_convrnn_accuracy = {'delay_0_ms': 91.4, 'delay_60_ms': 93.2, 'delay_100_ms': 95.1, 'delay_200_ms': 97.4, 'delay_300_ms': 98.8, 'delay_400_ms': 99.5}
sliding_ann_accuracy = {'delay_0_ms': 81.6, 'delay_60_ms': 85.3, 'delay_100_ms': 86.7, 'delay_200_ms': 91.5, 'delay_300_ms': 96.0, 'delay_400_ms': 98.7}
sliding_gru_accuracy = {'delay_0_ms': 86.3, 'delay_60_ms': 90.1, 'delay_100_ms': 91.4, 'delay_200_ms': 95.4, 'delay_300_ms': 98.0, 'delay_400_ms': 98.8}
model_accuracy = [sliding_ann_accuracy, sliding_gru_accuracy, raw_cnn_accuracy, raw_convrnn_accuracy]


## random losing channels
channel_random_lost_5 = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
channel_random_lost_6 = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
channel_random_lost_10 = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
channel_random_lost_15 = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
random_losing_accuracy = [raw_convrnn_accuracy, channel_random_lost_5, channel_random_lost_6, channel_random_lost_10, channel_random_lost_15]


## random recovery channels
channel_random_lost_5_recovered = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
channel_random_lost_6_recovered = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
channel_random_lost_10_recovered = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
channel_random_lost_15_recovered = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
random_recovery_accuracy = [raw_convrnn_accuracy, channel_random_lost_5_recovered, channel_random_lost_6_recovered, channel_random_lost_10_recovered, channel_random_lost_15_recovered]


## corner losing channels
channel_corner_lost_5 = {'delay_0_ms': 64.1, 'delay_60_ms': 69.0, 'delay_100_ms': 69.7, 'delay_200_ms': 71.6, 'delay_300_ms': 76.5, 'delay_400_ms': 81.3}
channel_corner_lost_6 = {'delay_0_ms': 65.1, 'delay_60_ms': 68.0, 'delay_100_ms': 73.7, 'delay_200_ms': 78, 'delay_300_ms': 83.5, 'delay_400_ms': 87.3}
channel_corner_lost_10 = {'delay_0_ms': 64.1, 'delay_60_ms': 69.0, 'delay_100_ms': 70.7, 'delay_200_ms': 72.6, 'delay_300_ms': 78.5, 'delay_400_ms': 82.3}
channel_corner_lost_15 = {'delay_0_ms': 55.1, 'delay_60_ms': 52.0, 'delay_100_ms': 50.7, 'delay_200_ms': 51.6, 'delay_300_ms': 58.5, 'delay_400_ms': 62.3}
corner_losing_accuracy = [raw_convrnn_accuracy, channel_corner_lost_5, channel_corner_lost_6, channel_corner_lost_10, channel_corner_lost_15]


## corner recovery channels
channel_corner_lost_5_recovered = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 93.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.2, 'delay_400_ms': 99.3}
channel_corner_lost_6_recovered = {'delay_0_ms': 88.1, 'delay_60_ms': 91.0, 'delay_100_ms': 93.7, 'delay_200_ms': 95.6, 'delay_300_ms': 98.9, 'delay_400_ms': 99.5}
channel_corner_lost_10_recovered = {'delay_0_ms': 86.1, 'delay_60_ms': 89.6, 'delay_100_ms': 90.9, 'delay_200_ms': 95.6, 'delay_300_ms': 98.0, 'delay_400_ms': 99.3}
channel_corner_lost_15_recovered = {'delay_0_ms': 75.1, 'delay_60_ms': 80.0, 'delay_100_ms': 81.7, 'delay_200_ms': 87.6, 'delay_300_ms': 94.5, 'delay_400_ms': 96.3}
corner_recovery_accuracy = [raw_convrnn_accuracy, channel_corner_lost_5_recovered, channel_corner_lost_6_recovered, channel_corner_lost_10_recovered, channel_corner_lost_15_recovered]


## plot the accuracy for each model at different delay points
import matplotlib.pyplot as plt
import numpy as np
accuracy = corner_recovery_accuracy
# Extract the x and y values from the dictionaries
x_values = list(accuracy[0].keys())  # Assuming all dictionaries have the same keys
y_values = np.array([[d[key] for key in x_values] for d in accuracy])

# Calculate the bar width and positions
n_bars = len(accuracy)
bar_width = 0.8 / n_bars
bar_positions = np.arange(len(x_values))

# Plot the data
colors = ['steelblue', 'wheat', 'darkorange', 'yellowgreen', 'pink', 'darkslateblue']
for i in range(n_bars):
    plt.bar(bar_positions + i * bar_width, y_values[i], width=bar_width, color=colors[i % len(colors)])

plt.xticks(bar_positions + bar_width * n_bars / 2, x_values)
plt.xlabel('Prediction time delay(ms)')
plt.ylabel('Prediction accuracy(%)')
plt.title('Prediction accuracy at different delay points')
# plt.legend(['Conventional features + Majority vote', 'Conventional features + GRU output', 'Instantaneous data + Majority vote', 'Instantaneous data + GRU output'])
# plt.legend(['channel_corner_lost_0', 'channel_corner_lost_5', 'channel_corner_lost_6', 'channel_corner_lost_10', 'channel_corner_lost_15'])
plt.legend(['channel_corner_lost_0_recovered','channel_corner_lost_5_recovered', 'channel_corner_lost_6_recovered', 'channel_corner_lost_10_recovered', 'channel_corner_lost_15_recovered'])
# Set y-axis limits and ticks
y_limit = 50
plt.ylim(y_limit, 100)
yticks = np.arange(y_limit, 102, 2)
plt.yticks(yticks)
plt.show()
