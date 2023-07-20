import numpy as np
from Generative_Model.Functions import GAN_Testing


## normalize data using min-max way
def normalizeEmgData(original_data, limit=2000):
    def normalizeMinMax(array):  # normalize the value to [-1, 1]
        min_val = -limit
        max_val = limit
        normalized_array = 2 * ((array - min_val) / (max_val - min_val)) - 1
        return normalized_array
    normalized_data = {}
    for key, arrays_list in original_data.items():
        clipped_arrays_list = [np.clip(array, -limit, limit) for array in arrays_list]
        normalized_data[key] = normalizeMinMax(np.vstack(clipped_arrays_list))
    return normalized_data


## generate and reorganize fake data
def generateFakeEmg(gan_models, real_emg_images, start_before_toeoff_ms, endtime_after_toeoff_ms):
    # generate fake data
    batch_size = 8192
    fake_old_data = {}
    generator_model = GAN_Testing.ModelTesting(gan_models['gen_BA'], batch_size)
    for transition_type, transition_value in real_emg_images.items():
        real_new_data = np.vstack(transition_value)  # size [n_samples, n_channel, length, width]
        fake_old_data[transition_type] = generator_model.testModel(real_new_data)

    # reorganize generated data
    fake_old_emg = {}
    for transition_type, transition_value in fake_old_data.items():
        transposed_emg = np.transpose(transition_value, (0, 2, 3, 1))
        reshaped_emg = np.reshape(transposed_emg, newshape=(transposed_emg.shape[0], -1), order='F')
        split_data = np.split(reshaped_emg, transposed_emg.shape[0] // (start_before_toeoff_ms + endtime_after_toeoff_ms))
        fake_old_emg[transition_type] = split_data

    return fake_old_emg


## calculate the average results from all classification models
def getAverageResults(overall_accuracy, overall_cm_recall):
    # Get the keys from the first dictionary (assuming all dicts have the same keys)
    keys = overall_accuracy[0].keys()
    # Create a new dictionary to store the averages
    average_accuracy = {}
    # Iterate over the keys
    for key in keys:
        # Calculate the average value for the current key
        average_value = np.mean([d[key] for d in overall_accuracy])
        # Add the average value to the average_dict
        average_accuracy[key] = average_value

    # Create a new dictionary to store the averages
    average_cm_recall = {}
    # Iterate over the keys
    for key in keys:
        # Stack the arrays along a new axis (axis 0)
        stacked_arrays = np.stack([d[key] for d in overall_cm_recall])
        # Calculate the average value for the current key
        average_array = np.mean(stacked_arrays, axis=0)
        # Add the average value to the average_dict
        average_cm_recall[key] = average_array
    return average_accuracy, average_cm_recall

