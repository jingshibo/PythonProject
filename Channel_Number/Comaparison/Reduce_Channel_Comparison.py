##
import matplotlib.pyplot as plt
import numpy as np

## model accuracy results
raw_cnn_accuracy = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
raw_convrnn_accuracy = {'delay_0_ms': 91.4, 'delay_60_ms': 93.2, 'delay_100_ms': 95.1, 'delay_200_ms': 97.4, 'delay_300_ms': 98.8, 'delay_400_ms': 99.5}
sliding_ann_accuracy = {'delay_0_ms': 81.6, 'delay_60_ms': 85.3, 'delay_100_ms': 86.7, 'delay_200_ms': 91.5, 'delay_300_ms': 96.0, 'delay_400_ms': 98.7}
sliding_gru_accuracy = {'delay_0_ms': 86.3, 'delay_60_ms': 90.1, 'delay_100_ms': 91.4, 'delay_200_ms': 95.4, 'delay_300_ms': 98.0, 'delay_400_ms': 98.8}
model_accuracy = [sliding_ann_accuracy, sliding_gru_accuracy, raw_cnn_accuracy, raw_convrnn_accuracy]


## reduce area results
channel_area_35 = {'delay_0_ms': 88.8, 'delay_60_ms': 91.1, 'delay_100_ms': 93.3, 'delay_200_ms': 94.8, 'delay_300_ms': 97.8, 'delay_400_ms': 99.0}
channel_area_25 = {'delay_0_ms': 87.7, 'delay_60_ms': 91.0, 'delay_100_ms': 92.7, 'delay_200_ms': 94.6, 'delay_300_ms': 97.5, 'delay_400_ms': 99.1}
channel_area_15 = {'delay_0_ms': 84.6, 'delay_60_ms': 87.5, 'delay_100_ms': 89.5, 'delay_200_ms': 93.2, 'delay_300_ms': 97.7, 'delay_400_ms': 99.3}
channel_area_6 = {'delay_0_ms': 80.8, 'delay_60_ms': 82.6, 'delay_100_ms': 85.7, 'delay_200_ms': 89.4, 'delay_300_ms': 94.5, 'delay_400_ms': 96.3}
channel_area_2 = {'delay_0_ms': 78.1, 'delay_60_ms': 82.0, 'delay_100_ms': 84.3, 'delay_200_ms': 88.3, 'delay_300_ms': 90.6, 'delay_400_ms': 94.3}
reduce_area_accuracy = [raw_convrnn_accuracy, channel_area_35, channel_area_25, channel_area_15, channel_area_6, channel_area_2]


## reduce density results
channel_density_33 = {'delay_0_ms': 89.6, 'delay_60_ms': 92.0, 'delay_100_ms': 92.7, 'delay_200_ms': 94.6, 'delay_300_ms': 98.1, 'delay_400_ms': 99.5}
channel_density_21 = {'delay_0_ms': 88.4, 'delay_60_ms': 91.5, 'delay_100_ms': 93.3, 'delay_200_ms': 94.1, 'delay_300_ms': 98.0, 'delay_400_ms': 99.3}
channel_density_11 = {'delay_0_ms': 82.1, 'delay_60_ms': 86.0, 'delay_100_ms': 88.7, 'delay_200_ms': 93, 'delay_300_ms': 96.5, 'delay_400_ms': 97.3}
channel_density_8 = {'delay_0_ms': 82.1, 'delay_60_ms': 85.9, 'delay_100_ms': 87.7, 'delay_200_ms': 91, 'delay_300_ms': 95.5, 'delay_400_ms': 96.5}
channel_muscle_bipolar = {'delay_0_ms': 78.1, 'delay_60_ms': 82.0, 'delay_100_ms': 84.3, 'delay_200_ms': 88.3, 'delay_300_ms': 90.6, 'delay_400_ms': 94.3}
reduce_density_accuracy = [raw_convrnn_accuracy, channel_density_33, channel_density_21, channel_density_11, channel_density_8, channel_area_2]


## reduce muscle results
channel_muscle_hdemg1 = {'delay_0_ms': 87.3, 'delay_60_ms': 90.0, 'delay_100_ms': 90.7, 'delay_200_ms': 91.1, 'delay_300_ms': 96.5, 'delay_400_ms': 99.1}
channel_muscle_hdemg2 = {'delay_0_ms': 83.1, 'delay_60_ms': 89.0, 'delay_100_ms': 91.7, 'delay_200_ms': 95.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
channel_muscle_bipolar1 = {'delay_0_ms': 64.1, 'delay_60_ms': 66.0, 'delay_100_ms': 68.1, 'delay_200_ms': 70.6, 'delay_300_ms': 73.5, 'delay_400_ms': 78.3}
channel_muscle_bipolar2 = {'delay_0_ms': 71.1, 'delay_60_ms': 77.6, 'delay_100_ms': 81.7, 'delay_200_ms': 85.9, 'delay_300_ms': 89.5, 'delay_400_ms': 90.3}
reduce_muscle_accuracy = [raw_convrnn_accuracy, channel_muscle_hdemg1, channel_muscle_hdemg2, channel_area_2, channel_muscle_bipolar1, channel_muscle_bipolar2]


## plot the accuracy for each model at different delay points
accuracy = reduce_area_accuracy
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
plt.title('Prediction accuracy different delay points')
# plt.legend(['Conventional features + Majority vote', 'Conventional features + GRU output', 'Instantaneous data + Majority vote', 'Instantaneous data + GRU output'])
# plt.legend(['channel_area_all', 'channel_area_35', 'channel_area_25', 'channel_area_15', 'channel_area_6', 'channel_bipolar'])
# plt.legend(['channel_density_all', 'channel_density_33', 'channel_density_21', 'channel_density_11', 'channel_density_8', 'channel_bipolar'])
plt.legend(['hdemg both muscles', 'hdemg RF muscle', 'hdemg TA muscle', 'bipolar both muscles', 'bipolar RF muscle', 'bipolar TA muscle'])
# Set y-axis limits and ticks
y_limit = 60
plt.ylim(y_limit, 100)
yticks = np.arange(y_limit, 102, 2)
plt.yticks(yticks)
plt.show()


##


