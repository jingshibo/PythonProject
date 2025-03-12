##
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import copy


##  average accuracies across subjects
def average_bipolar_mix_subject(all_subjects):
    results = {}
    # Assuming all subjects have the same model types, use the first subject to get the model types
    bipolar_types = all_subjects[next(iter(all_subjects))].keys()

    for bipoalr in bipolar_types:
        accuracies = []
        cm_recalls = []
        cm_nums = []

        for subject in all_subjects.values():
            accuracies.append(subject[bipoalr]['accuracy'])
            cm_recalls.append(subject[bipoalr]['cm_recall'])
            cm_nums.append(subject[bipoalr]['cm_num'])

        # Calculate mean and std for accuracies
        accuracy_mean = np.mean(accuracies)
        accuracy_std = np.std(accuracies)

        # Calculate mean for cm_recall and cm_num
        cm_recall_mean = np.mean(np.stack(cm_recalls), axis=0)
        cm_num_mean = np.mean(np.stack(cm_nums), axis=0)

        results[bipoalr] = {
            'accuracy_mean': accuracy_mean,
            'accuracy_std': accuracy_std,
            'cm_recall_mean': cm_recall_mean,
            'cm_num_mean': cm_num_mean
        }

    return results


## group the results for the same muscle number together
def groupResultByMuscleNumber(average_by_bipolar, result_set):
    # Initialize an empty dictionary to store results based on '+' count
    results_by_category = {i: {} for i in range(6)}  # '6' refer to the number of bipolars
    # Iterate through the keys and aggregate based on '+' count
    for bipolar_type, metrics in average_by_bipolar.items():
        plus_count = bipolar_type.count('+')
        results_by_category[plus_count][bipolar_type] = copy.deepcopy(metrics)
    # adjustment

    return results_by_category


## claculate mean accuracy across the same number of bipolar EMG for each subject
def average_accuracy_by_subject(all_subjects):
    mean_metrics = {}

    for subject, data in all_subjects.items():
        # Dictionary to store metrics grouped by the number of '+' symbols
        grouped_metrics = {}

        for key, metrics in data.items():
            plus_count = key.count('+')

            if plus_count not in grouped_metrics:
                grouped_metrics[plus_count] = {metric_key: [] for metric_key in metrics.keys()}

            # Append the metrics to the corresponding group
            for metric_key, metric_value in metrics.items():
                grouped_metrics[plus_count][metric_key].append(metric_value)

        # Calculate the mean for each group
        mean_metrics[subject] = {}
        for plus_count, metrics in grouped_metrics.items():
            mean_metrics[subject][plus_count] = {}
            for metric_key, metric_values in metrics.items():
                if isinstance(metric_values[0], (list, np.ndarray)):  # Handle arrays/lists
                    mean_value = np.mean(metric_values, axis=0)
                else:  # Handle scalar values like accuracy
                    mean_value = np.mean(metric_values)

                mean_metrics[subject][plus_count][f'{metric_key}'] = mean_value

    return mean_metrics


## calculate paried t-test values between different bipolar numbers
def calculateTtestValues(average_by_subject, average_by_bipolar_group):
    max_plus_count = max(max(metrics.keys()) for metrics in average_by_subject.values())

    # Initialize a new dictionary to reorganize the data
    reorganized_dict = {i: {} for i in range(max_plus_count + 1)}  # Assuming 0 to 5 "+" symbols

    # Iterate over the existing structure and reorganize
    for subject, metrics in average_by_subject.items():
        for plus_count, data in metrics.items():
            reorganized_dict[plus_count][subject] = data['accuracy']

    # Initialize a list to store t-test results
    t_test_results = {}

    # Iterate over pairs of keywords (0 with 1, 1 with 2, etc.)
    for i in range(max_plus_count + 1):
        if i == 0:
            key1 = i
            key2 = i
        else:
            key1 = i
            key2 = i - 1

        # Extract the accuracy values into a list for the corresponding subjects
        accuracies1 = [reorganized_dict[key1][subject] for subject in reorganized_dict[key1]]
        accuracies2 = [reorganized_dict[key2][subject] for subject in reorganized_dict[key2]]

        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(accuracies1, accuracies2)

        # Store the result
        average_by_bipolar_group[key1]['ttest'] = p_value


## mean value of the six single bipolar EMG
def calculate_bipolar_group_mean(plus_aggregated_results):
    group_mean = {}

    for plus_count, data in plus_aggregated_results.items():
        # Calculate average and std for accuracies
        accuracies = data['accuracies']
        accuracy_mean = np.mean(accuracies)
        accuracy_std = np.std(accuracies)

        # Calculate mean for cm_recall and cm_num matrices
        cm_recall_mean = np.mean(np.stack(data['cm_recalls']), axis=0)
        cm_num_mean = np.mean(np.stack(data['cm_nums']), axis=0)

        group_mean[plus_count] = {
            'accuracy_mean': accuracy_mean,
            'accuracy_std': accuracy_std,
            'cm_recall_mean': cm_recall_mean,
            'cm_num_mean': cm_num_mean
        }

    return group_mean


## combine the results from all subjects together
def aggregate_results(all_subjects):
    # Initialize an empty dictionary to store aggregated results
    aggregated_results = {}

    # Assuming all subjects have the same model types, use the first subject to get the model types
    bipolar_types = all_subjects[next(iter(all_subjects))].keys()

    # Aggregate metrics for each model type
    for bipolar in bipolar_types:
        accuracies = [subject[bipolar]['accuracy'] for subject in all_subjects.values() if bipolar in subject]
        cm_recalls = [subject[bipolar]['cm_recall'] for subject in all_subjects.values() if bipolar in subject]
        cm_nums = [subject[bipolar]['cm_num'] for subject in all_subjects.values() if bipolar in subject]

        aggregated_results[bipolar] = {
            'accuracies': accuracies,
            'cm_recalls': cm_recalls,
            'cm_nums': cm_nums
        }

    return aggregated_results


## combine the results from the same number of bipolars together
def aggregate_by_bipolar_number(aggregated_results):
    # Initialize an empty dictionary to store results based on '+' count
    plus_aggregated_results = {i: {'accuracies': [], 'cm_recalls': [], 'cm_nums': []} for i in range(6)}  # '6' refer to the number of bipolars

    # Iterate through the keys and aggregate based on '+' count
    for bipolar_type, metrics in aggregated_results.items():
        plus_count = bipolar_type.count('+')
        plus_aggregated_results[plus_count]['accuracies'].extend(metrics['accuracies'])
        plus_aggregated_results[plus_count]['cm_recalls'].extend(metrics['cm_recalls'])
        plus_aggregated_results[plus_count]['cm_nums'].extend(metrics['cm_nums'])

    return plus_aggregated_results


## plot the box accuracy for each bipolar number
def plotBipolarBoxAccuracy(combined_by_bipolar, RF_accuracy_experiment, RF_accuracy_derived):
    # Extracting the accuracies for plotting
    data_to_plot = [metrics['accuracies'] for plus_count, metrics in sorted(combined_by_bipolar.items())]
    # Create labels for the x-axis based on plus_count
    labels = [f"{plus_count + 1}" for plus_count in sorted(combined_by_bipolar.keys())]

    # Create the box plot
    plt.figure(figsize=(10, 6))
    fontsize = 35
    linewidth = 2
    # mpl.rcParams['font.family'] = 'Times New Roman'

    box = plt.boxplot(data_to_plot, patch_artist=True,
        flierprops={'marker': '+', 'markeredgecolor': 'red', 'markersize': 20, 'markeredgewidth': linewidth},
        boxprops={'facecolor': 'None', 'edgecolor': 'blue', 'linewidth': linewidth},  # Increase the outline border width here
        whiskerprops={'linestyle': 'dashed', 'color': 'black', 'linewidth': linewidth, 'dashes': (5, 5)},
        medianprops={'color': 'red', 'linewidth': linewidth},
        capprops={'linewidth': linewidth},
        whis=2)

    # Setting the colors for the median lines
    for median in box['medians']:
        median.set_color('red')

    plt.xticks(range(1, len(labels) + 1), labels, fontsize=fontsize)
    plt.yticks(range(50, 105, 5), fontsize=fontsize)
    plt.xlabel('Number of Bipolar EMG', fontsize=fontsize)
    plt.ylabel('Classification Accuracy(%)', fontsize=fontsize)
    plt.title('')

    # Plot a horizontal line at the y-coordinate defined by rf_accuracy
    plt.axhline(y=RF_accuracy_experiment, color='green', linestyle='dashed', linewidth=2)
    plt.axhline(y=RF_accuracy_derived, color='darkorange', linestyle='dashed', linewidth=2)

    plt.show()

    # Extract median values
    median_values = [median.get_ydata()[0] for median in box['medians']]

    return median_values

## plot the box accuracy for the bipolar derived from HDsEMG
def plotOldBoxBipolar(bipolar_accuracy_from_hdsemg_matrix, RF_accuracy):
    # create labels
    labels = ['2', '4', '6', '8', '10', '12', '14', '16', '18', '64']

    # Create the box plot
    plt.figure(figsize=(10, 6))
    fontsize = 30
    linewidth = 2
    # mpl.rcParams['font.family'] = 'Times New Roman'

    box = plt.boxplot(bipolar_accuracy_from_hdsemg_matrix, patch_artist=True,
        flierprops={'marker': '+', 'markeredgecolor': 'red', 'markersize': 20, 'markeredgewidth': linewidth},
        boxprops={'facecolor': 'None', 'edgecolor': 'blue', 'linewidth': linewidth},  # Increase the outline border width here
        whiskerprops={'linestyle': 'dashed', 'color': 'black', 'linewidth': linewidth, 'dashes': (5, 5)},
        medianprops={'color': 'red', 'linewidth': linewidth},
        capprops={'linewidth': linewidth},
        whis=1.5)

    # Setting the colors for the median lines
    for median in box['medians']:
        median.set_color('red')

    plt.xticks(range(1, len(labels) + 1), labels, fontsize=fontsize)
    plt.yticks(range(50, 105, 5), fontsize=fontsize)
    plt.xlabel('Number of Electrodes', fontsize=fontsize)
    plt.ylabel('Classification Accuracy(%)', fontsize=fontsize)
    plt.title('')

    # # Plot a horizontal line at the y-coordinate defined by rf_accuracy
    # plt.axhline(y=RF_accuracy, color='green', linestyle='dashed', linewidth=1.5)

    plt.show()


## plot average values for each bipolar number group
def plotMeanAccuracy(average_by_bipolar_group, results_by_category, hdsemg_accuracy, derived_12_accuracy):
    # Assuming metrics_summary is already calculated and available
    accuracy_means = [data['accuracy_mean'] for data in average_by_bipolar_group.values()]
    accuracy_stds = [data['accuracy_std'] for data in average_by_bipolar_group.values()]
    plus_counts = list(average_by_bipolar_group.keys())
    font_size = 35

    # Creating the plot
    plt.figure(figsize=(10, 6))
    # mpl.rcParams['font.family'] = 'Times New Roman'

    # Plotting accuracy means with error bars for standard deviation
    bars = plt.bar(plus_counts, accuracy_means, yerr=accuracy_stds, capsize=5, color='yellowgreen', width=0.5)

    # Annotating the bars with t-test p-values by drawing horizontal lines between consecutive bars
    for i in range(1, len(plus_counts)):
        key1 = plus_counts[i-1]
        key2 = plus_counts[i]
        p_value = average_by_bipolar_group[key2].get('ttest', None)

        if p_value is not None:
            # Determine x and y coordinates for the lines
            x_left = i - 1  # left bar
            x_right = i  # right bar
            y_max = max(accuracy_means[i-1] + accuracy_stds[i-1], accuracy_means[i] + accuracy_stds[i])
            y, h, col = y_max + 1, 1, 'black'  # Adjust `h` to control the height of horizontal lines

            # Plotting the horizontal line (without vertical lines at the ends)
            plt.plot([x_left, x_right], [y+h, y+h], lw=1.5, c=col)

            # Add significance stars or "ns" based on p-value
            if p_value < 0.01:
                plt.text((x_left + x_right) * 0.5, y+h-0.5, "**", ha='center', va='bottom', color='r', fontsize=font_size*1)
            elif p_value < 0.05:
                plt.text((x_left + x_right) * 0.5, y+h-0.5, "*", ha='center', va='bottom', color='r', fontsize=font_size*1)
            else:
                plt.text((x_left + x_right) * 0.5, y+h, "ns", ha='center', va='bottom', color='r', fontsize=font_size*0.8)

    # Set font size for labels and title
    plt.xlabel('Number of Bipolar EMG', fontsize=font_size)
    plt.ylabel('Classification Accuracy(%)', fontsize=font_size)
    plt.title('', fontsize=font_size)

    # Set font size for tick labels
    plt.xticks(plus_counts, [f'{count + 1}' for count in plus_counts], fontsize=font_size)

    # Set the y-axis limits to 60 to 100
    plt.ylim(60, 105)  # Adjust the upper limit if necessary
    plt.yticks(np.arange(60, 101, 5), fontsize=font_size)

    # Add grid and display the plot
    # plt.grid(axis='y', zorder=0, alpha=1)

    # Plot horizontal reference lines
    plt.axhline(y=hdsemg_accuracy, color='red', linestyle='dashed', linewidth=2)
    plt.axhline(y=derived_12_accuracy, color='blue', linestyle='dashed', linewidth=2)

    # Plot average results for each muscle combination
    x_labels = []
    y_values = []
    for key, muscle_groups in results_by_category.items():
        for muscle_group, metrics in muscle_groups.items():
            x_labels.append(key)
            y_values.append(metrics['accuracy_mean'])

    # Creating the scatter plot
    plt.scatter(x_labels, y_values, s=75)

    # Remove the outline border of the plot
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    plt.show()
