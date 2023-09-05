'''

'''

import numpy as np
import copy
from Cycle_GAN.Functions import Data_Processing
import random
from scipy import signal

## separate the dataset into multiple timepoint bins
def separateByInterval(data, timepoint_interval=50, length=850, output_list=False):
    '''
    :param data: input emg dataset
    :param timepoint_interval: how large is the bin to separate the dataset
    :param length: how long is each repetition per gait mode in the experiment
    :param output_list: data structure of the output, whether a list of a single ndarray for each timepoint
    :return: separated dataset
    '''

    separated_result = {}
    # iterate over the data in chunks of size period`
    for i in range(0, data.shape[0], length):
        period_data = data[i:i + length]  # the same result as data[i:i+period,:,:,:]
        # for each chunk, create `timepoint` number of keys
        for j in range(length // timepoint_interval):
            timepoint_data = period_data[j * timepoint_interval:(j + 1) * timepoint_interval]
            key = f"timepoint_{j * timepoint_interval}"
            # If the key exists, concatenate the data, else just assign the data
            if output_list:  # put the data of the same time point from multiple experiment sessions into a list.
                if key in separated_result:
                    separated_result[key].append(timepoint_data)
                else:
                    separated_result[key] = [timepoint_data]
            else:  # concat the data of the same time point from multiple experiment sessions into a single ndarray
                if key in separated_result:
                    separated_result[key] = np.concatenate([separated_result[key], timepoint_data], axis=0)
                else:
                    separated_result[key] = timepoint_data
    return separated_result


## Generate fake data based on defined index pair correspondence between gen_data_1 and gen_data_2.
def generateFakeData(reorganized_data, interval, repetition=1, random_pairing=True):
    # Extracting the main data structures from the input dictionary
    gen_data_1 = reorganized_data['gen_data_1']
    gen_data_2 = reorganized_data['gen_data_2']
    blending_factors = reorganized_data['blending_factors']

    # List to store multiple sets of new data
    fake_data = []

    # Repeat the generation process multiple times
    for _ in range(repetition):  # Note: if random_pairing=False, only 1 repetition is valid
        # Dictionary to store the newly generated data
        new_data_dict = {}
        # Define the random pairing relationship for indices once; this relationship will be applied to all timepoints in this repetition.
        num_samples = gen_data_1['timepoint_0'].shape[0]
        if random_pairing:
            paired_indices = np.random.permutation(num_samples)  # shuffle the experiment session order for gen_data_2
        else:
            paired_indices = range(num_samples)  # use the original experiment session order for pairing

        # Iterate over each timepoint key
        for time_point in gen_data_1.keys():
            # Determine the corresponding factor key in blending_factors
            timepoint_number = int(time_point.split('_')[-1])
            factor_key = f"timepoint_{(timepoint_number // interval) * interval}"
            factor = blending_factors[factor_key]

            # Fetch data from gen_data_1 and its paired index from gen_data_2 based on the predetermined pairing
            samples_1 = gen_data_1[time_point]  # note: this is not only one sample, samples_1 is an array of [sample, channel, height, width]
            samples_2 = gen_data_2[time_point][paired_indices]

            # Apply the blending formula to produce the new data
            if factor.shape[1] == 1:  # one alpha parameter
                new_data = samples_1 * factor + samples_2 * (1 - factor)
            elif factor.shape[1] == 2:  # two alpha parameters
                new_data = samples_1 * factor[:, 0, :, :][:, np.newaxis, :, :] + samples_2 * factor[:, 1, :, :][:, np.newaxis, :, :]
            elif factor.shape[1] == 3:  # three alpha parameters
                new_data = samples_1 * factor[:, 0, :, :][:, np.newaxis, :, :] + samples_2 * factor[:, 1, :, :][:, np.newaxis, :,
                :] + factor[:, 2, :, :][:, np.newaxis, :, :]
            new_data_dict[time_point] = new_data
        # Append the generated data to the data_list
        fake_data.append(new_data_dict)

    return fake_data


## Generate fake data based on combinations of every two complete emg curves gen_data_1 and gen_data_2.
def generateFakeDataByCurve(reorganized_data, interval, repetition=1, random_pairing=True):
    '''
    This function is modified based on the input and output of the above generateFakeData() function in order to be compatible with other
    functions, so there are some additional data transformations and input arguments that look unnecessary.
    '''
    def reshapeGenData(data):
        # Get the shape parameters
        num_timepoints = len(data)
        num_sample, channels, num_row, num_col = data['timepoint_0'].shape

        # Pre-allocate a NumPy array for the reshaped data
        reshaped_data = [np.zeros((num_timepoints, channels, num_row, num_col)) for _ in range(num_sample)]
        # Loop through the sorted keys and fill the pre-allocated NumPy arrays
        for idx, key in enumerate(sorted(data.keys(), key=lambda x: int(x.split('_')[-1]))):
            current_array = data[key]  # Shape: (num_sample, 1, num_row, num_col)
            for i in range(num_sample):
                reshaped_data[i][idx] = current_array[i]
        return reshaped_data

    def reshapeBlendingFactors(data):
        sorted_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[-1]))
        # Calculate the constant interval once, using the first two keys
        interval = int(sorted_keys[1].split('_')[-1]) - int(sorted_keys[0].split('_')[-1])

        # Initialize an empty list to store the reorganized data
        reshaped_data_list = []
        # Loop through the sorted keys to reorganize the data
        for key in sorted_keys:
            current_array = data[key]  # Shape: (1, 2, 13, 10)
            # Replicate the current array 'constant_interval' times
            replicated_array = np.repeat(current_array, interval, axis=0)  # Shape will be (constant_interval, 2, 13, 10)
            # Add to the reorganized data list
            reshaped_data_list.append(replicated_array)
        # Concatenate all the arrays to form a single 4D array
        reshaped_data = np.concatenate(reshaped_data_list, axis=0)
        return reshaped_data

    # Function to combine gen_data_1 and gen_data_2 based on blending_factors
    def performBlendingOperation(gen_data_1, gen_data_2, blending_factors):
        # Determine the shape parameters
        num_samples = len(gen_data_1) * len(gen_data_2)
        array_shape = gen_data_1[0].shape

        # Pre-allocate a NumPy array to store the combined data
        generated_data_array = np.zeros((num_samples, *array_shape))
        # Pre-compute the slices of blending_factors to avoid repeated computation
        blend_slices = [blending_factors[:, i, :, :][:, np.newaxis, :, :] for i in range(blending_factors.shape[1])]
        # Loop through each array in gen_data_1 and gen_data_2 to combine them
        index = 0
        for array1 in gen_data_1:
            for array2 in gen_data_2:
                if blending_factors.shape[1] == 1:  # One alpha parameter
                    generated_data_array[index] = array1 * blend_slices[0] + array2 * (1 - blend_slices[0])
                elif blending_factors.shape[1] == 2:  # Two alpha parameters
                    generated_data_array[index] = array1 * blend_slices[0] + array2 * blend_slices[1]
                elif blending_factors.shape[1] == 3:  # Three alpha parameters
                    generated_data_array[index] = array1 * blend_slices[0] + array2 * blend_slices[1] + blend_slices[2]
                index += 1
        # Convert the combined_data_array into a list of individual samples
        combined_data_list = [generated_data_array[i] for i in range(num_samples)]
        return combined_data_list

    # Extracting the main data structures from the input dictionary
    gen_data_1 = reshapeGenData(reorganized_data['gen_data_1'])
    gen_data_2 = reshapeGenData(reorganized_data['gen_data_2'])
    blending_factors = reshapeBlendingFactors(reorganized_data['blending_factors'])

    # Initialize a dictionary to store the reorganized data
    fake_data = {}
    generated_data_list = performBlendingOperation(gen_data_1, gen_data_2, blending_factors)
    # Loop through each timepoint (0 to 849)
    for timepoint in range(len(reorganized_data['gen_data_1'])):
        # Extract the slice corresponding to the current timepoint from each of the 121 arrays
        slices_at_timepoint = [array[timepoint] for array in generated_data_list]
        # Stack these slices along a new axis to form a new array of shape (121, 1, 13, 10)
        array_at_timepoint = np.stack(slices_at_timepoint)
        # Add this array to the dictionary with the key corresponding to the current timepoint
        fake_data[f"timepoint_{timepoint}"] = array_at_timepoint

    return [fake_data]


## Reorganize the generated_data in fake_data_list to construct time series data form at each repetition and reshape them for classification
def reorganizeFakeData(fake_data_list):
    reorganized_fake_data = []
    # Loop through each generated_data dictionary in fake_data_list
    for generated_data in fake_data_list:
        # Get the shape parameters dynamically
        num_samples, channels, num_row, num_col = generated_data['timepoint_0'].shape
        num_timepoints = len(generated_data)
        # Pre-allocate a NumPy array for the reorganized data
        reorganized_array = np.zeros((num_samples, num_timepoints, channels, num_row, num_col))
        # Ensure the keys are sorted correctly
        sorted_keys = sorted(generated_data.keys(), key=lambda x: int(x.split('_')[-1]))

        # Fill in the pre-allocated NumPy array
        for t, key in enumerate(sorted_keys):
            reorganized_array[:, t] = generated_data[key]  # reorganized_array[:, t] are equivalent to reorganized_array[:, t, :, :, :].
        # Split the reorganized_array into individual sample series and remove the unnecessary first dimension
        reorganized_fake_data.extend([sample_series.squeeze(axis=0) for sample_series in np.split(reorganized_array, num_samples, axis=0)])

    # reshape generated fake data for classification
    reshaped_fake_data = []
    for value in reorganized_fake_data:
        transposed_emg = np.transpose(value, (0, 2, 3, 1))
        reshaped_emg = np.reshape(transposed_emg, newshape=(transposed_emg.shape[0], -1), order='F')
        reshaped_fake_data.append(reshaped_emg)

    return reshaped_fake_data


## substitute original emg data using generated fake data
def replaceUsingFakeEmg(fake_data, real_emg_normalized):
    synthetic_data = copy.deepcopy(real_emg_normalized)
    for fake_type, fake_value in fake_data.items():
        synthetic_data[fake_type] = fake_value
    return synthetic_data


## substitute the test dataset using real data
def replaceUsingRealEmg(mode_to_substitute, real_emg_normalized, train_dataset, test_ratio):
    test_dataset = copy.deepcopy(train_dataset)
    real_value = real_emg_normalized[mode_to_substitute]
    random.Random(5).shuffle(real_value)
    test_dataset['group_0']['test_set'][mode_to_substitute] = real_value[0: int(len(real_value) * test_ratio)]
    return test_dataset


## filtering
def smoothGeneratedData(data_list, cutoff_frequency):
    sos = signal.butter(4, cutoff_frequency, fs=1000, btype="lowpass", output='sos')
    filtered_fake_data = [signal.sosfiltfilt(sos, data, axis=0) for data in data_list]
    return filtered_fake_data

