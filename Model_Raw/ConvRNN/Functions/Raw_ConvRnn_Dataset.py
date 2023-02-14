##
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import copy

##  combine data of all gait events into a single dataset
def combineNormalizedDataset(sliding_window_dataset):
    normalized_groups = {}
    for group_number, group_value in sliding_window_dataset.items():
        # initialize training set and test set for each group
        train_feature_x = []
        train_feature_y = []
        test_feature_x = []
        test_feature_y = []
        for set_type, set_value in group_value.items():
            if set_type == 'train_set':  # combine all data into a dataset
                for transition_label, transition_data in set_value.items():
                    train_feature_x.extend(transition_data)
                    train_feature_y.extend([transition_label] * len(transition_data))
            elif set_type == 'test_set':  # keep the structure unchanged
                for transition_label, transition_data in set_value.items():
                    test_feature_x.extend(transition_data)
                    test_feature_y.extend([transition_label] * len(transition_data))

        # mean and std value for the training group
        mean_x = np.mean(np.concatenate(train_feature_x, axis=-1), axis=-1)[:, :, np.newaxis]
        std_x = np.std(np.concatenate(train_feature_x, axis=-1), axis=-1)[:, :, np.newaxis]
        for train_repetition_number, train_repetition_value in enumerate(train_feature_x):
            train_feature_x[train_repetition_number] = (train_repetition_value - mean_x) / std_x
        for test_repetition_number, test_repetition_value in enumerate(test_feature_x):
            test_feature_x[test_repetition_number] = (test_repetition_value - mean_x) / std_x

        # one-hot encode categories (according to the alphabetical order)
        train_int_y = LabelEncoder().fit_transform(train_feature_y)
        train_onehot_y = tf.keras.utils.to_categorical(train_int_y)
        test_int_y = LabelEncoder().fit_transform(test_feature_y)
        test_onehot_y = tf.keras.utils.to_categorical(test_int_y)
        # put training data and test data into one group
        normalized_groups[group_number] = {"train_feature_x": train_feature_x, "train_int_y": train_int_y, "train_onehot_y": train_onehot_y,
            "test_feature_x": test_feature_x, "test_int_y": test_int_y, "test_onehot_y": test_onehot_y}

    return normalized_groups


##  construct gru dataset
def shuffleTrainingSet(normalized_groups, predict_window_per_repetition, predict_window_shift_unit, predict_of_window_number):
    shuffled_groups = copy.deepcopy(normalized_groups)  # # use reference instead of copy to save memory

    # combine the data from all repetitions into one single list
    for group_number, group_value in shuffled_groups.items():
        for key, value in group_value.items():
            if key == 'train_feature_x' or key == 'test_feature_x':
                for repetition_number, repetition_value in enumerate(value):
                    repetition_data = []
                    for i in range(0, predict_window_per_repetition, predict_window_shift_unit):
                        repetition_data.append(repetition_value[:, :, i:i+predict_of_window_number+1][:, :, np.newaxis, :])
                    value[repetition_number] = repetition_data
                group_value[key] = [item for sublist in value for item in sublist]  # convert nested list to a single list
            elif key == 'train_int_y' or key == 'test_int_y' or key == 'train_onehot_y' or key == 'test_onehot_y':
                group_value[key] = np.repeat(value, predict_window_per_repetition, axis=0)  # repeat y to match the number of x

        # Shuffles the indices
        def shuffle_list_using_reference(my_list, reference_list):  # shuffle a list using another list as a reference
            combined_list = list(zip(reference_list, my_list))
            combined_list.sort()
            shuffled_list = [x[1] for x in combined_list]
            return shuffled_list

        data_number = len(group_value['train_feature_x'])
        idx = np.arange(data_number)
        np.random.shuffle(idx)
        train_idx = idx[: int(data_number)].tolist()

        # shuffle the training data
        group_value['train_feature_x'] = shuffle_list_using_reference(group_value['train_feature_x'], train_idx)
        group_value['train_int_y'] = np.array(shuffle_list_using_reference(group_value['train_int_y'].tolist(), train_idx))
        group_value['train_onehot_y'] = np.array(shuffle_list_using_reference([row for row in group_value['train_onehot_y']], train_idx))

    return shuffled_groups