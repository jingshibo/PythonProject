'''

'''


import numpy as np
import copy
from Cycle_GAN.Functions import Data_Processing
import random


## separate the dataset into multiple timepoint bins
def separateByTimeInterval(data, timepoint_interval=50, length=850, output_list=False):
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
    new_data_list = []

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
        new_data_list.append(new_data_dict)

    return new_data_list


## Reorganize the generated_data in fake_data_list to construct time series data form at each repetition and reshape them for classification
def reorganizeFakeData(fake_data_list):
    reorganized_fake_data = []
    # Iterate over each new_data_dict in data_list
    for generated_data in fake_data_list:
        # number of samples generated per timepoint
        num_samples = generated_data['timepoint_0'].shape[0]
        # Initialize a temporary list to store the reorganized data for the current new_data_dict
        temp_reorganized_list = [[] for _ in range(num_samples)]

        # Ensure the keys are sorted correctly based on the numeric value after "timepoint_"
        sorted_keys = sorted(generated_data.keys(), key=lambda x: int(x.split('_')[-1]))
        # Iterate through the keys (timepoints)
        for key in sorted_keys:  # the sorted() function sorts the keys in ascending order.
            timepoint_data = generated_data[key]

            # The enumerate function will return two values on each iteration: an index idx and the value at that index sample
            for idx, sample in enumerate(timepoint_data):  # enumerate will iterate over the first axis
                temp_reorganized_list[idx].append(sample)

        # Convert inner lists to numpy arrays and append to the reorganized_data_list
        reorganized_fake_data.extend([np.array(sample_series) for sample_series in temp_reorganized_list])

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


