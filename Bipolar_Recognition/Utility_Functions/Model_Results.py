import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import datetime


## a single CNN model
def classifyUsingCnnModel(shuffled_groups):
    '''
    A basic 2-layer CNN model
    '''

    models = []
    results = []

    for group_number, group_value in shuffled_groups.items():
        # input data
        train_set_x = group_value['train_feature_x']  # data_format:'channels_last'
        train_set_y = group_value['train_onehot_y']
        test_set_x = group_value['test_feature_x']  # data_format:'channels_last'
        test_set_y = group_value['test_onehot_y']
        class_number = len(set(group_value['train_int_y']))

        # layer parameters
        regularization = tf.keras.regularizers.L2(0.0001)
        initializer = tf.keras.initializers.HeNormal()
        # model structure
        inputs = tf.keras.Input(shape=(train_set_x.shape[1], train_set_x.shape[2], train_set_x.shape[3]))
        x = tf.keras.layers.Conv2D(30, 5, dilation_rate=2, data_format='channels_last')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
        x = tf.keras.layers.Flatten(data_format='channels_last')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(class_number, kernel_regularizer=regularization)(x)
        outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_cnn")
        # view models
        model.summary()

        # model parameters
        num_epochs = 60
        decay_epochs = 20
        batch_size = 512
        decay_steps = decay_epochs * len(train_set_y) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.005, decay_steps=decay_steps, decay_rate=0.3)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-08)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
        # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

        # train model
        now = datetime.datetime.now()
        model.fit(train_set_x, train_set_y, validation_split=0.1, epochs=num_epochs, batch_size=batch_size, shuffle=True, verbose='auto')
        print(datetime.datetime.now() - now)

        # test model
        predictions = model.predict(test_set_x)  # return predicted probabilities
        predict_y = np.argmax(predictions, axis=-1)  # return predicted labels
        test_loss, test_accuracy = model.evaluate(test_set_x, test_set_y)  # return loss and accuracy values

        results.append({"true_value": group_value['test_int_y'], "predict_softmax": predictions, "predict_value": predict_y,
            "predict_accuracy": test_accuracy})
        models.append(model)

    return models, results


## save classification accuracy and cm recall values
def saveResult(subject, average_accuracies, average_cm_numbers, average_cm_recalls, model_type, result_set, project='HDsEMG_Recognition'):
    data_dir = f'D:\Data\{project}\subject_{subject}\\results'
    result_file = f'subject_{subject}_model_{model_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    # Combine the two dictionaries into one
    combined_results = {'accuracy': average_accuracies, 'cm_num': [arr.tolist() for arr in average_cm_numbers],
        'cm_recall': [arr.tolist() for arr in average_cm_recalls]}

    # Save to JSON file
    with open(result_path, 'w') as f:
        json.dump(combined_results, f, indent=8)


## read classification accuracy and cm recall values
def loadResult(subject, model_type, result_set, project='HDsEMG_Recognition'):
    data_dir = f'D:\Data\{project}\subject_{subject}\\results'
    result_file = f'subject_{subject}_model_{model_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    with open(result_path, 'r') as f:
        loaded_data = json.load(f)

    combined_results = {'accuracy': loaded_data['accuracy'], 'cm_recall': [np.array(lst) for lst in loaded_data['cm_recall']],
        'cm_num': [np.array(lst) for lst in loaded_data['cm_num']]}
    return combined_results


## Calculates the mean values across subjects
def averageAcrossSubjects(data_dict):
    # Automatically determine the number of indices
    num_indices = len(next(iter(data_dict.values())))

    # Accumulate values for each index
    accumulators = [[] for _ in range(num_indices)]
    for data_list in data_dict.values():
        for i in range(num_indices):
            accumulators[i].append(data_list[i])

    # Calculate mean for each index
    mean_values = [np.mean(acc, axis=0) for acc in accumulators]
    return mean_values


## plot accuracy values for all subjects
def averageHdsemgResults(all_subjects):
    # Extract hdsemg_0 from all subjects
    hdsemg_results = {subject_key: subject_data['hdsemg_0'] for subject_key, subject_data in all_subjects.items()}

    # Extract accuracy and cm_recall from hdsemg_results
    hdsemg_accuracy = {subject_key: subject_data['accuracy'] for subject_key, subject_data in hdsemg_results.items()}
    hdsemg_cm_recall = {subject_key: subject_data['cm_recall'] for subject_key, subject_data in hdsemg_results.items()}

    # Calculate the mean accuracies
    mean_accuracies = averageAcrossSubjects(hdsemg_accuracy)
    hdsemg_accuracy['average'] = mean_accuracies

    # Calculate the mean recall values
    mean_cm_recall = averageAcrossSubjects(hdsemg_cm_recall)
    hdsemg_cm_recall['average'] = mean_cm_recall

    return hdsemg_results, hdsemg_accuracy, hdsemg_cm_recall


## calculate the mean accuracy for each number of electrodes across all subjects
def averageBipolarAccuracies(bipolar_accuracy):
    # Initialize a dictionary to store the mean accuracy for each condition
    mean_accuracies = {}

    # Get the list of conditions based on one of the subjects
    conditions = list(bipolar_accuracy['Number1'].keys())

    # Calculate the mean accuracy for each condition across all subjects
    for condition in conditions:
        accuracies = []
        for subject in bipolar_accuracy:
            accuracies.append(bipolar_accuracy[subject][condition]['accuracy'])

        # Compute the mean accuracy for the current condition
        mean_accuracies[condition] = np.mean(accuracies)

    return mean_accuracies


## plot HDsEMG accuracy values for each subject individually
def plotHdsemgAccuracy(hdsemg_accuracy):
    # Set global font size for the plot
    plt.rcParams.update({'font.size': 30})

    # Assuming hdsemg_accuracy is your dictionary
    subjects = list(hdsemg_accuracy.keys())  # Get the list of subjects, including 'average'
    subjects.remove('average')  # Remove 'average' from the sorting process

    # Sort subjects by accuracy at index 0
    subjects.sort(key=lambda subject: hdsemg_accuracy[subject][0])

    # Add 'average' back at the end of the list
    subjects.append('average')

    indices = [0, 1, 2]  # The three accuracy indices
    bar_width = 0.25  # Width of the bars
    colors = ['yellowgreen', 'pink', 'steelblue']  # Colors for the bars

    # Custom legend labels
    legend_labels = ['n=5', 'n=6', 'n=7']

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # X positions for the groups of bars, including space for the average
    x = np.arange(len(subjects))

    # Plot bars for each index with custom colors and labels
    for i, (idx, color) in enumerate(zip(indices, colors)):
        accuracies = [hdsemg_accuracy[subject][idx] for subject in subjects]
        ax.bar(x + i * bar_width, accuracies, bar_width, label=legend_labels[i], color=color)

    # Set y-axis limits
    ax.set_ylim(93, 100)

    # Add labels and title
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Classification Accuracy (%)')
    xtick_labels = [f'S{i + 1}' for i in range(len(subjects) - 1)] + ['Average']
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.legend()  # Display the custom legend

    # Show the plot
    plt.tight_layout()

    # Add grid and display the plot
    plt.grid(axis='y', zorder=0, alpha=1)
    plt.show()

    # Remove the outline border of the plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


## plot the box accuracy for the bipolar derived from HDsEMG
def plotDerivedBipolarBox(bipolar_accuracy):
    # Extracting accuracy values for each condition
    conditions = ['bipolar_1_0', 'bipolar_2_0', 'bipolar_3_0', 'bipolar_4_0', 'bipolar_5_0', 'bipolar_6_0', 'bipolar_7_0', 'bipolar_8_0',
        'bipolar_9_0', 'hdsemg_0']

    # Custom labels for the x-axis
    labels = ['2', '4', '6', '8', '10', '12', '14', '16', '18', '64']

    # Prepare data for boxplot
    accuracy_data = []
    for condition in conditions:
        accuracies = [bipolar_accuracy[subject][condition]['accuracy'] for subject in bipolar_accuracy]
        accuracy_data.append(accuracies)

    # Customization parameters
    fontsize = 35
    linewidth = 2

    # Create the box plot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(accuracy_data, patch_artist=True,
        flierprops={'marker': '+', 'markeredgecolor': 'red', 'markersize': 20, 'markeredgewidth': linewidth},
        boxprops={'facecolor': 'None', 'edgecolor': 'blue', 'linewidth': linewidth},
        whiskerprops={'linestyle': 'dashed', 'color': 'black', 'linewidth': linewidth, 'dashes': (5, 5)},
        medianprops={'color': 'red', 'linewidth': linewidth}, capprops={'linewidth': linewidth}, whis=1.5)

    # Setting the colors for the median lines
    for median in box['medians']:
        median.set_color('red')

    # Set the x-axis and y-axis labels, ticks, and titles
    plt.xticks(range(1, len(labels) + 1), labels, fontsize=fontsize)
    plt.yticks(range(50, 105, 5), fontsize=fontsize)
    plt.xlabel('Number of Electrodes', fontsize=fontsize)
    plt.ylabel('Classification Accuracy (%)', fontsize=fontsize)
    plt.title('')

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Extract median values
    median_values = [median.get_ydata()[0] for median in box['medians']]

    return median_values

## calculate mean and std values across subject under the electrode shift scenarios
def calcuShiftMeanStd(shift_accuracy):
    # Initialize a dictionary to store the mean and std values
    mean_std_accuracy = {}

    # Get the list of conditions (assuming all subjects have the same set of conditions)
    conditions = list(shift_accuracy['Number1'].keys())

    # Iterate over each condition
    for condition in conditions:
        # Extract accuracy values across all subjects for the current condition
        accuracies = [shift_accuracy[subject][condition]['accuracy'] for subject in shift_accuracy]

        # Calculate mean and standard deviation
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # Store the mean and std in the dictionary
        mean_std_accuracy[condition] = {'mean': mean_accuracy, 'std': std_accuracy}

    # Print the result to verify
    return mean_std_accuracy


## plot the accuracy of derived bipolar EMG under electrode shift
def plotShiftBipolarAccuracy(shift_mean_std):
    # Set global font size for the plot
    plt.rcParams.update({'font.size': 20})

    # Extract keys for the original, horizontal, and vertical shifts
    original_key = 'bipolar_original_0'
    h_keys = [key for key in shift_mean_std if 'bipolar_h' in key]
    v_keys = [key for key in shift_mean_std if 'bipolar_v' in key]

    # Ensure that h_keys and v_keys are aligned based on electrode numbers
    h_keys.sort()
    v_keys.sort()

    # Extract mean and std values for the original, horizontal, and vertical shifts
    original_mean = shift_mean_std[original_key]['mean']
    original_std = shift_mean_std[original_key]['std']

    h_means = [shift_mean_std[key]['mean'] for key in h_keys]
    h_stds = [shift_mean_std[key]['std'] for key in h_keys]
    v_means = [shift_mean_std[key]['mean'] for key in v_keys]
    v_stds = [shift_mean_std[key]['std'] for key in v_keys]

    # Extract electrode numbers for labels (after the original)
    electrodes = [key.split('_')[3] for key in h_keys]

    # Set up bar positions
    bar_width = 0.35
    index = np.arange(len(electrodes) + 1)  # +1 to account for the original

    # Plotting the bar chart
    plt.figure(figsize=(12, 8))

    # Adjust the position of the original bar to be centered
    plt.bar(index[0] + bar_width / 2, original_mean, bar_width, yerr=original_std, capsize=5, label='Original Position', color='yellowgreen')

    # Plot the grouped bars for h and v shifts
    bar1 = plt.bar(index[1:], h_means, bar_width, yerr=h_stds, capsize=5, label='Horizontal Shift', color='pink')
    bar2 = plt.bar(index[1:] + bar_width, v_means, bar_width, yerr=v_stds, capsize=5, label='Vertical Shift', color='steelblue')

    # Adding titles and labels
    plt.title('')
    plt.xlabel('Number of Bipolar Shift')
    plt.ylabel('Classification Accuracy (%)')

    # Set x-axis labels, starting with '0' for the original and followed by electrode numbers
    x_labels = ['0'] + electrodes
    plt.xticks(index + bar_width / 2, x_labels)
    plt.ylim(20, 100)

    # Add legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    # Add grid and display the plot
    plt.grid(axis='y', zorder=0, alpha=1)
    plt.show()

    # Remove the outline border of the plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)



## plot the accuracy of HDsEMG under electrode shift and recovery
def plotShiftHdsemgAccuracy(shift_mean_std):
    # Set global font size for the plot
    plt.rcParams.update({'font.size': 20})

    # Extract the relevant keys
    keys_h_left = ['hdsemg_h_shift_0', 'aug_left_0']
    keys_v_up = ['hdsemg_v_shift_0', 'aug_up_0']
    original_key = 'hdsemg_original_0'

    # Extract mean and std values for the keys
    original_mean = shift_mean_std[original_key]['mean']
    original_std = shift_mean_std[original_key]['std']

    means_h_left = [shift_mean_std[key]['mean'] for key in keys_h_left]
    stds_h_left = [shift_mean_std[key]['std'] for key in keys_h_left]
    means_v_up = [shift_mean_std[key]['mean'] for key in keys_v_up]
    stds_v_up = [shift_mean_std[key]['std'] for key in keys_v_up]

    # Define bar width and positions
    bar_width = 0.35
    index = np.arange(2)  # Since we have 2 pairs: H-Shift & Left, V-Shift & Up

    # Plot the bar chart
    plt.figure(figsize=(10, 6))

    # Plot original as a single bar centered at the first tick
    plt.bar(0, original_mean, bar_width, yerr=original_std, capsize=5, color='yellowgreen', label='Original Position')

    # Plot H-Shift and Left as adjacent bars
    plt.bar(index + 1, means_h_left, bar_width, yerr=stds_h_left, capsize=5, color='pink', label='Horizontal Shift')

    # Plot V-Shift and Up as adjacent bars
    plt.bar(index + 1 + bar_width, means_v_up, bar_width, yerr=stds_v_up, capsize=5, color='steelblue', label='Vertical Shift')

    # Adding titles and labels
    plt.title('')
    plt.xlabel('HDsEMG Shift and Recovery')
    plt.ylabel('Classification Accuracy (%)')

    # Set the custom x-ticks with appropriate labels
    x_labels = ['Original Position', 'Electrode Shift', 'Data Augmentation']
    plt.xticks([0, 1 + bar_width / 2, 2 + bar_width / 2], x_labels)
    plt.ylim(20, 100)

    # Add legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    # Add grid and display the plot
    plt.grid(axis='y', zorder=0, alpha=1)
    plt.show()

    # Remove the outline border of the plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


## plot the accuracy of both bipolar EMG and HDsEMG under electrode shift in the same chart
def plotElectrodeShiftResults(shift_mean_std):
    # Set global font size for the plot
    plt.rcParams.update({'font.size': 34})

    # Data preparation based on the specified order
    # Bipolar data
    bipolar_original_key = 'bipolar_original_0'
    bipolar_h_keys = ['bipolar_h_shift_1_0', 'bipolar_h_shift_3_0', 'bipolar_h_shift_4_0', 'bipolar_h_shift_6_0']
    bipolar_v_keys = ['bipolar_v_shift_1_0', 'bipolar_v_shift_3_0', 'bipolar_v_shift_4_0', 'bipolar_v_shift_6_0']

    # Non-bipolar data
    hdsemg_original_key = 'hdsemg_original_0'
    hdsemg_h_key = 'hdsemg_h_shift_0'
    hdsemg_v_key = 'hdsemg_v_shift_0'
    aug_left_key = 'aug_both_left_0'
    aug_up_key = 'aug_both_up_0'

    # Extract mean and std for bipolar original
    bipolar_original_mean = shift_mean_std[bipolar_original_key]['mean']
    bipolar_original_std = shift_mean_std[bipolar_original_key]['std']

    # Extract means and stds for other keys
    bipolar_means_h = [shift_mean_std[key]['mean'] for key in bipolar_h_keys]
    bipolar_stds_h = [shift_mean_std[key]['std'] for key in bipolar_h_keys]
    bipolar_means_v = [shift_mean_std[key]['mean'] for key in bipolar_v_keys]
    bipolar_stds_v = [shift_mean_std[key]['std'] for key in bipolar_v_keys]

    hdsemg_original_mean = shift_mean_std[hdsemg_original_key]['mean']
    hdsemg_original_std = shift_mean_std[hdsemg_original_key]['std']

    hdsemg_means = [shift_mean_std[hdsemg_h_key]['mean'], shift_mean_std[hdsemg_v_key]['mean']]
    hdsemg_stds = [shift_mean_std[hdsemg_h_key]['std'], shift_mean_std[hdsemg_v_key]['std']]

    aug_means = [shift_mean_std[aug_left_key]['mean'], shift_mean_std[aug_up_key]['mean']]
    aug_stds = [shift_mean_std[aug_left_key]['std'], shift_mean_std[aug_up_key]['std']]

    # Define bar width and positions
    bar_width = 0.35
    index = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # Positions for the group ticks

    # Plotting the bar chart
    plt.figure(figsize=(14, 8))

    # Plot bipolar original as a single bar centered at tick 1
    plt.bar(index[0] + bar_width / 2, bipolar_original_mean, bar_width, yerr=bipolar_original_std, capsize=5, color='yellowgreen',
        label='Original')

    # Plot bipolar h and v shifts in pairs at ticks 2 to 5
    for i in range(4):
        plt.bar(index[i + 1], bipolar_means_h[i], bar_width, yerr=bipolar_stds_h[i], capsize=5, color='pink')
        plt.bar(index[i + 1] + bar_width, bipolar_means_v[i], bar_width, yerr=bipolar_stds_v[i], capsize=5, color='steelblue')

    # Plot hdsemg original as a single bar centered at tick 6
    plt.bar(index[5] + bar_width / 2, hdsemg_original_mean, bar_width, yerr=hdsemg_original_std, capsize=5, color='yellowgreen')

    # Plot hdsemg h and v shifts in pairs at tick 7
    plt.bar(index[6], hdsemg_means[0], bar_width, yerr=hdsemg_stds[0], capsize=5, color='pink')
    plt.bar(index[6] + bar_width, hdsemg_means[1], bar_width, yerr=hdsemg_stds[1], capsize=5, color='steelblue')

    # Plot aug left and up in pairs at tick 8
    plt.bar(index[7], aug_means[0], bar_width, yerr=aug_stds[0], capsize=5, color='pink')
    plt.bar(index[7] + bar_width, aug_means[1], bar_width, yerr=aug_stds[1], capsize=5, color='steelblue')

    # Adding titles and labels
    plt.title('')
    plt.xlabel('Electrode Shift Conditions')
    plt.ylabel('Classification Accuracy (%)')

    # Set the custom x-ticks with appropriate labels
    x_labels = ['Original\nBipolar ', '1 Bipolar\nShift', '3 Bipolar\nShift', '4 Bipolar\nShift', '6 Bipolar\nShift', 'Original\nHDsEMG',
        'HDsEMG\nShift', 'Data Aug-\nmentation']
    plt.xticks(index + bar_width / 2, x_labels, rotation=0, ha='center')
    plt.ylim(20, 100)
    plt.yticks(np.arange(20, 101, 10))

    # Display a simplified legend
    plt.legend(['Original Position', 'Transversal Shift', 'Longitudinal Shift'], loc='upper center', bbox_to_anchor=(0.5, 1.15))

    # Display the plot
    plt.tight_layout()
    # Add grid and display the plot
    plt.grid(axis='y', zorder=0, alpha=1)
    plt.show()

    # Remove the outline border of the plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)