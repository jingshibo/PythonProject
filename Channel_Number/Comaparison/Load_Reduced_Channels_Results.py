## import
from Channel_Number.Functions import Results_Organization
from Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt

all_subjects = {}  # save all subject results


##
subject = 'Number4'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)

##
subject = 'Number5'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)

##
subject = 'Shibo'
version = 1
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)

##
subject = 'zehao'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)

##
subject = 'Number3'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)

##
subject = 'Number2'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)

##
subject = 'Number1'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)

##
subject = 'Number0'
version = 0
result_set = 0
all_subjects[subject] = []
get_results = Results_Organization.getSubjectResults(subject, version)
all_subjects[subject] = get_results.getAllResults(result_set)


##  print accuracy results at each 100 delay
import copy
extract_delay = ['delay_0_ms', 'delay_60_ms', 'delay_100_ms', 'delay_200_ms', 'delay_300_ms', 'delay_400_ms']
extracted_results = copy.deepcopy(all_subjects)
for subject_number, subject_results in extracted_results.items():
    for dataset, datavalue in subject_results.items():
        for condition, results in datavalue.items():
            for item, value in results.items():
                results[item] = {k: value[k] for k in extract_delay}


##  combine the results across different subject for each extract_delay key
import pandas as pd
combined_results = {}
# Iterate over all the keys in the extracted_results dictionary
for subject_number, subject_results in extracted_results.items():
    for dataset, datavalue in subject_results.items():
        for condition, results in datavalue.items():
            for item, value in results.items():
                # If the dataset key is not already in the combined_results dictionary, create a new dictionary for it
                if dataset not in combined_results:
                    combined_results[dataset] = {}
                # If the condition key is not already in the dataset dictionary, create a new dictionary for it
                if condition not in combined_results[dataset]:
                    combined_results[dataset][condition] = {}
                # If the item key is not already in the condition dictionary, create a new dictionary for it
                if item not in combined_results[dataset][condition]:
                    combined_results[dataset][condition][item] = {}
                # Iterate over each key-value pair in the results dictionary
                for extract_delay_key, extract_delay_value in value.items():
                    # If the extract_delay key is not already in the item dictionary, create a new list for it
                    if extract_delay_key not in combined_results[dataset][condition][item]:
                        combined_results[dataset][condition][item][extract_delay_key] = []
                    # Add the extract_delay value for this item across all subject_number to the corresponding list
                    combined_results[dataset][condition][item][extract_delay_key].append(extract_delay_value)


## calculate statistical results for each condition
import pandas as pd
from scipy.stats import ttest_rel

mean_std_value = copy.deepcopy(combined_results)
for dataset, datavalue in combined_results.items():
    for condition, results in datavalue.items():
        for item, value in results.items():
            if item == 'accuracy':
                mean_std_value[dataset][condition][item] = pd.DataFrame(value)
                mean_std_value[dataset][condition]['mean'] = pd.DataFrame(value).mean()
                mean_std_value[dataset][condition]['std'] = pd.DataFrame(value).std()
                mean_std_value[dataset][condition]['ttest'] = {}
                for col in mean_std_value[dataset][condition][item].columns:
                    t_stat, p_value = ttest_rel(mean_std_value[dataset][condition][item][col], combined_results['model_type']['Raw_ConvRnn']['accuracy'][col])
                    mean_std_value[dataset][condition]['ttest'][col] = p_value
                mean_std_value[dataset][condition]['ttest'] = pd.Series(mean_std_value[dataset][condition]['ttest'])


## reorganize mean, std and ttest values
# Define a list of keys to extract from each dictionary
keys = ['mean', 'std', 'ttest']

# Create a new dictionary to store the reorganized results
reorganized_results = copy.deepcopy(mean_std_value)
# Iterate through each dataset and extract the specified keys
for dataset, datavalue in mean_std_value.items():
    # Create new keys in the dict for storage
    for key in keys:
        reorganized_results[dataset][key] = {}

    # put hdemg results into all other dataset for comparison, excluding the 'model_type' dataset
    if dataset != 'model_type':
        for key in keys:
            reorganized_results[dataset][key]['hdemg'] = mean_std_value['model_type']['Raw_ConvRnn'][key]
    # extract the statistical results from each condition to the dataset dict it belongs to
    for condition, results in datavalue.items():
        for key in keys:
            reorganized_results[dataset][key][condition] = results[key]
    # put bipoalr results into reduce dataset for comparison,excluding the 'model_type' dataset
    if dataset in ['reduce_density_dataset', 'reduce_muscle_dataset'] and dataset != 'model_type':
        for key in keys:
            reorganized_results[dataset][key]['bipolar'] = mean_std_value['reduce_area_dataset']['channel_area_2'][key]

    # convert each dictionary to a DataFrame
    for key in keys:
        reorganized_results[dataset][key] = pd.DataFrame(reorganized_results[dataset][key])


## plot mean and std figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create sample data
mean_df = reorganized_results['reduce_area_dataset']['mean']
std_df = reorganized_results['reduce_area_dataset']['std']

# Create color list
color_list = ['steelblue', 'wheat', 'darkorange', 'yellowgreen', 'pink', 'darkgray']
# Plot bar chart with error bars
ax = mean_df.plot.bar(yerr=std_df, capsize=4, width=0.8, color=color_list)

# add_significance(ax, mean_df.values, std_df.values)
# Set x-axis label
x_label = ax.set_xlabel('Prediction time delay(ms)')
# Rotate x-axis label
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
# Set y-axis label
ax.set_ylabel('Prediction accuracy(%)')
# Set plot title
ax.set_title('Prediction accuracy different delay points')
# Set y-axis limit
ax.set_ylim([60, 102])
# Set legend position
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
# gets the current figure instance and assigns it to the variable fig
fig = plt.gcf()
# sets the size of the figure to be 6 inches by 6 inches
fig.set_size_inches(8, 6)
# adjusts the spacing between the subplots in the figure
fig.subplots_adjust(right=0.8)
# Show plot
plt.show()


##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# create example dataframes
# df_mean = pd.DataFrame({'A': [10, 20, 30, 40], 'B': [15, 25, 35, 45], 'C': [18, 28, 38, 48]})
# df_std = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 3, 4, 5], 'C': [3, 4, 5, 6]})
# df_pval = pd.DataFrame({'A': [np.nan, np.nan, np.nan, np.nan], 'B': [0.05, 0.01, 0.1, 0.005], 'C': [0.05, 0.01, 0.1, 0.005]})

# Create sample data
df_mean = reorganized_results['reduce_area_dataset']['mean']
df_std = reorganized_results['reduce_area_dataset']['std']
df_pval = reorganized_results['reduce_area_dataset']['ttest']


# plot the mean values
ax = df_mean.plot(kind='bar', yerr=df_std, capsize=5)
bar_width = ax.patches[0].get_width()  # assume all bars have the same width

# iterate over the columns and rows of df_pval
for j in range(df_pval.shape[1]):
    for i in range(df_pval.shape[0]):
        pval = df_pval.iloc[i, j]
        if not pd.isna(pval):
            # if pval < 0.1:  # only plot for pval < 0.1
            # calculate the x and y coordinates of the horizontal lines
            x_left, y_left = ax.patches[i].get_x() + bar_width / 2, df_mean.iloc[i, 0]
            x_right, y_right = ax.patches[j * df_pval.shape[0] + i].get_x() + bar_width / 2, df_mean.iloc[i, j]

            # calculate the height and y coordinate of the significance stars
            height = max(df_mean.iloc[i, :])
            star_height = height + max(df_std.iloc[i, :]) * 0.6 * j
            y_left, y_right = star_height, star_height

            # add the line and significance stars to the plot
            ax.plot([x_left, x_right], [y_left, y_right], 'k-', lw=1)
            if pval < 0.01:
                ax.text((x_left + x_right) * 0.5, star_height, "***", ha='center', va='bottom', color='r', fontsize=15)
            elif pval < 0.05:
                ax.text((x_left + x_right) * 0.5, star_height, "**", ha='center', va='bottom', color='r', fontsize=15)
            elif pval < 0.1:
                ax.text((x_left + x_right) * 0.5, star_height, "*", ha='center', va='bottom', color='r', fontsize=15)
            else:  # if not significant
                ax.text((x_left + x_right) * 0.5, star_height, "ns", ha='center', va='bottom', color='r', fontsize=12)

            # add small vertical bars at two ends of the lines
            left_line = ax.plot([x_left, x_left], [y_left - 0.01 * height, y_left], 'k-', lw=1)
            right_line = ax.plot([x_right, x_right], [y_right - 0.01 * height, y_right], 'k-', lw=1)

# add x and y axis labels
plt.xlabel('Columns')
plt.ylabel('Mean Values')

# show the plot
plt.show()


##
import pandas as pd
import matplotlib.pyplot as plt

# create some sample data
data = {'A': [10, 20, 30], 'B': [20, 30, 40], 'C': [30, 40, 50], 'D': [40, 50, 60]}
df = pd.DataFrame.from_dict(data)

# create the bar chart
ax = df.plot(kind='bar')

# obtain the x-coordinate of each bar
bar_width = ax.patches[0].get_width()  # assume all bars have the same width
for i, patch in enumerate(ax.patches):
    x = patch.get_x() + (i % 3) * bar_width
    print(f"X-coordinate of bar {i+1}: {x}")

# obtain the x-coordinate of each bar
for i, patch in enumerate(ax.patches):
    x = patch.get_x() + patch.get_width() / 2
    print(f"X-coordinate of bar {i+1}: {x}")

##
import matplotlib.pyplot as plt

# create some sample data
x = ['A', 'B', 'C', 'D']
y_left = [10, 20, 30, 40]
y_right = [15, 25, 35, 45]

# create the bar chart
fig, ax = plt.subplots()
bar_width = 0.35
x_indexes = list(range(len(x)))
ax.bar(x_indexes, y_left, width=bar_width, label='Column 1')
ax.bar([i + bar_width for i in x_indexes], y_right, width=bar_width, label='Column 2')

# set the x-axis labels
ax.set_xticks(x_indexes)
ax.set_xticklabels(x)

# add a legend
ax.legend()
for i, patch in enumerate(ax.patches):
    x = patch.get_x() + (i % 3) * bar_width
    print(f"X-coordinate of bar {i+1}: {x}")
# show the chart
plt.show()


##

