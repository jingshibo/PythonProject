import numpy as np
from Generative_Model.Functions import GAN_Testing
from sklearn.metrics import confusion_matrix
import copy

## min-max normalization method
def normalizeMinMax(array, limit):  # normalize the value to [-1, 1]
    min_val = -limit
    max_val = limit
    normalized_array = 2 * ((array - min_val) / (max_val - min_val)) - 1
    return normalized_array

## normalize data using min-max way
def normalizeEmgData(original_data, limit=2000):
    normalized_data = {}
    for locomotion_type, locomotion_value in original_data.items():
        clipped_arrays_list = [np.clip(array, -limit, limit) for array in locomotion_value]
        normalized_data[locomotion_type] = normalizeMinMax(np.vstack(clipped_arrays_list), limit)
    return normalized_data

## generate and reorganize fake data
def generateFakeEmg(gan_model, real_emg_data, start_before_toeoff_ms, endtime_after_toeoff_ms, batch_size):
    # generate fake data
    fake_old_data = {}
    generator_model = GAN_Testing.ModelTesting(gan_model, batch_size)
    for transition_type, transition_value in real_emg_data.items():
        print(transition_type, ':')
        fake_old_data[transition_type] = generator_model.testModel(transition_value)

    # reorganize generated data
    fake_old_emg = {}
    for transition_type, transition_value in fake_old_data.items():
        transposed_emg = np.transpose(transition_value, (0, 2, 3, 1))
        reshaped_emg = np.reshape(transposed_emg, newshape=(transposed_emg.shape[0], -1), order='F')
        split_data = np.split(reshaped_emg, transposed_emg.shape[0] // (start_before_toeoff_ms + endtime_after_toeoff_ms))
        fake_old_emg[transition_type] = split_data

    return fake_old_emg

## substitute generated emg using real emg
def substituteFakeImages(fake_emg, real_emg_preprocessed, limit, emg_NOT_to_substitute='all'):
    generated_emg = copy.deepcopy(fake_emg)
    if emg_NOT_to_substitute == 'all':
        return generated_emg
    else:
        for locomotion_type, locomotion_value in generated_emg.items():
            if locomotion_type not in emg_NOT_to_substitute:   # which transition types to be substituted
                real_values = real_emg_preprocessed[locomotion_type]
                normalized_arrays_list = [normalizeMinMax(np.clip(array, -limit, limit), limit) for array in real_values]
                generated_emg[locomotion_type] = normalized_arrays_list
                print(locomotion_type)
        return generated_emg

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


## calculate average accuracy and cm based on majority vote results
def slidingMvResults(test_results):
    accuracy_allgroup = []
    cm_allgroup = []
    for each_group in test_results:
        true_y = each_group['true_value']
        predict_y = each_group['predict_value']
        numCorrect = np.count_nonzero(true_y == predict_y, axis=0)
        accuracy_allgroup.append(numCorrect / true_y.shape[0] * 100)
        # loop over the results at each delay point
        delay_cm = []
        for i in range(0, true_y.shape[1]):
            delay_cm.append(
                confusion_matrix(y_true=true_y[:, i], y_pred=predict_y[:, i]))  # each confusion matrix in the list belongs to a delay point
        cm_allgroup.append(np.array(delay_cm))
    return accuracy_allgroup, cm_allgroup


## add delay information to calculated average accuracy and cm reall values
def averageAccuracyCm(accuracy_allgroup, cm_allgroup, feature_window_increment_ms, predict_window_shift_unit):
    average_accuracy = np.mean(np.array(accuracy_allgroup), axis=0)
    average_accuracy_with_delay = {f'delay_{delay * feature_window_increment_ms * predict_window_shift_unit}_ms': value for delay, value in
        enumerate(average_accuracy.tolist())}

    sum_cm = np.sum(np.array(cm_allgroup), axis=0)
    average_cm_recall = {}
    for delay in range(sum_cm.shape[0]):
        cm = sum_cm[delay, :, :]
        cm_recall = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3)  # calculate cm recall
        average_cm_recall[f'delay_{delay * feature_window_increment_ms * predict_window_shift_unit}_ms'] = cm_recall
    return average_accuracy_with_delay, average_cm_recall

