
## accuracy results
raw_cnn_accuracy = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.6, 'delay_300_ms': 98.5, 'delay_400_ms': 99.3}
raw_convrnn_accuracy = {'delay_0_ms': 91.4, 'delay_60_ms': 93.2, 'delay_100_ms': 95.1, 'delay_200_ms': 97.4, 'delay_300_ms': 98.8, 'delay_400_ms': 99.5}
sliding_ann_accuracy = {'delay_0_ms': 81.6, 'delay_60_ms': 85.3, 'delay_100_ms': 86.7, 'delay_200_ms': 91.5, 'delay_300_ms': 96.0, 'delay_400_ms': 98.7}
sliding_gru_accuracy = {'delay_0_ms': 86.3, 'delay_60_ms': 90.1, 'delay_100_ms': 91.4, 'delay_200_ms': 95.4, 'delay_300_ms': 98.0, 'delay_400_ms': 98.8}
accuracy = [sliding_ann_accuracy, sliding_gru_accuracy, raw_cnn_accuracy, raw_convrnn_accuracy]


## plot the accuracy for each model at different delay points
import matplotlib.pyplot as plt
import numpy as np
# Extract the x and y values from the dictionaries
x_values = list(accuracy[0].keys())  # Assuming all dictionaries have the same keys
y_values = np.array([[d[key] for key in x_values] for d in accuracy])

# Calculate the bar width and positions
n_bars = len(accuracy)
bar_width = 0.8 / n_bars
bar_positions = np.arange(len(x_values))

# Plot the data
colors = ['steelblue', 'wheat', 'darkorange', 'yellowgreen']
for i in range(n_bars):
    plt.bar(bar_positions + i * bar_width, y_values[i], width=bar_width, color=colors[i % len(colors)])

plt.xticks(bar_positions + bar_width * n_bars / 2, x_values)
plt.xlabel('Prediction time delay(ms)')
plt.ylabel('Prediction accuracy(%)')
plt.title('Prediction accuracy for each model at different delay points')
plt.legend(['Conventional features + Majority vote', 'Conventional features + GRU output', 'Instantaneous data + Majority vote', 'Instantaneous data + GRU output'])
# Set y-axis limits and ticks
plt.ylim(80, 100)
yticks = np.arange(80, 102, 2)
plt.yticks(yticks)
plt.show()

##

