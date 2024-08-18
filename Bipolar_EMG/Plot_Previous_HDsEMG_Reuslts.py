##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl


## HDsEMG results from the previous paper
accuracy_dict = {'n=5': [95.6, 96.8, 97.5, 97.8, 98.3, 98.8, 99.2, 97.7],
 'n=6': [96.8, 98.3, 98.2, 97.9, 98.8, 99.4, 99.7, 98.4],
 'n=7': [97.1, 98.6, 99.5, 98.3, 99.4, 99.4, 99.8, 98.9]}

std_dict = {'n=5': [0.7, 0.4, 0.35, 0.19, 0.19, 0.22, 0.35, 0.36],
            'n=6': [0.55, 0.2, 0.17, 0.33, 0.35, 0.08, 0.05, 0.25],
            'n=7': [0.5, 0.2, 0.17, 0.16, 0.35, 0.08, 0.05, 0.22]}


## plot the accuracy
# Converting the dictionary to a DataFrame
accuracy_df = pd.DataFrame(accuracy_dict)
accuracy_df.index = [f'Subject {i+1}' if i < len(accuracy_df) - 1 else 'Average' for i in range(len(accuracy_df))]

mpl.rcParams['font.family'] = 'Times New Roman'
font_size = 30
# Plotting using DataFrame's plot function
accuracy_df.plot(yerr=std_dict, capsize=4, kind='bar', figsize=(12, 8), ylim=(94, 100), width=0.75, color=['yellowgreen', 'pink', 'steelblue'])

# Adding details to the plot
plt.xlabel('Subject Number', fontsize=font_size)
plt.ylabel('Classification Accuracy (%)', fontsize=font_size)
plt.title('')
plt.xticks(rotation=0, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.legend(fontsize=font_size)
# Add grid and display the plot
plt.grid(axis='y', zorder=0, alpha=1)
plt.show()

# Remove the outline border of the plot
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)


## HDsEMG derived shift results from the previous paper
accuracy_shift = [95.2, 87.0, 48, 40.8, 29.4, 96.2, 32.4, 93.9]
std_shift = [0.8, 3, 8, 5, 1.2, 0.4, 4, 0.65]

# Creating a DataFrame for easy plotting
accuracy_shift_df = pd.DataFrame({'Accuracy Shift': accuracy_shift})
accuracy_shift_df.index = ['Original\nBipolar', '1 Bipolar\nShift', '3 Bipolar\nShift', '4 Bipolar\nShift', '6 Bipolar\nShift',
 'Original\nHDsEMG', 'HDsEMG\nShift', 'Data\nAugmentation']

# Plotting the accuracy shift as a bar chart
accuracy_shift_df.plot.bar(yerr=std_shift, figsize=(10, 6), capsize=4, color='steelblue', width=0.5, ylim=(20, 100), legend=False)

# Adding details to the plot
plt.xlabel('Number of Bipolar Shift', fontsize=font_size)
plt.ylabel('Classification Accuracy (%)', fontsize=font_size)
plt.title('')
plt.xticks(rotation=0, fontsize=font_size)
plt.yticks(fontsize=font_size)
# Add grid and display the plot
plt.grid(axis='y', zorder=0, alpha=1)
# Adjusting the bottom margin to ensure 'Type' label is visible
plt.subplots_adjust(bottom=0.2)
plt.show()

# Remove the outline border of the plot
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
##

