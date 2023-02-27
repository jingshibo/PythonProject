## import
import copy
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


##  seperate data from each repetitions using sliding feature windows
def separateEmgData(cross_validation_groups, feature_window_size, increment=64):
    sliding_window_dataset = copy.deepcopy(cross_validation_groups)
    feature_window_per_repetition = int((cross_validation_groups['group_0']['train_set']['emg_LWLW'][0].shape[0] - feature_window_size) / increment + 1)
    for group_number, group_data in cross_validation_groups.items():
        for set_type, set_value in group_data.items():
            for transition_label, transition_data in set_value.items():
                for repetition_number, repetition_data in enumerate(transition_data):
                    windows_per_repetition = []
                    # if window_size=512, increment=64, the sample number is 25 per repetition
                    for i in range(0, repetition_data.shape[0] - feature_window_size + 1, increment):
                        windows_per_repetition.append(repetition_data[i:i + feature_window_size, :])
                    sliding_window_dataset[group_number][set_type][transition_label][repetition_number] = np.transpose(
                        np.array(windows_per_repetition).astype(np.float32), (1, 2, 0))
    return sliding_window_dataset, feature_window_per_repetition


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
                    train_feature_y.extend([transition_label] * len(transition_data) * transition_data[0].shape[-1])
            elif set_type == 'test_set':  # keep the structure unchanged
                for transition_label, transition_data in set_value.items():
                    test_feature_x.extend(transition_data)
                    test_feature_y.extend([transition_label] * len(transition_data) * transition_data[0].shape[-1])
        sliding_window_dataset[group_number] = []  # delete data to release the memory

        # normalization
        mean_x = np.mean(np.concatenate(train_feature_x, axis=-1), axis=-1)[:, :, np.newaxis]
        std_x = np.std(np.concatenate(train_feature_x, axis=-1), axis=-1)[:, :, np.newaxis]
        train_feature_x = (np.concatenate(train_feature_x, axis=-1) - mean_x) / std_x
        test_feature_x = (np.concatenate(test_feature_x, axis=-1) - mean_x) / std_x

        # one-hot encode categories (according to the alphabetical order)
        train_int_y = LabelEncoder().fit_transform(train_feature_y)
        train_onehot_y = tf.keras.utils.to_categorical(train_int_y)
        test_int_y = LabelEncoder().fit_transform(test_feature_y)
        test_onehot_y = tf.keras.utils.to_categorical(test_int_y)

        # put training data and test data into one group
        normalized_groups[group_number] = {"train_feature_x": train_feature_x, "train_int_y": train_int_y, "train_onehot_y": train_onehot_y,
            "test_feature_x": test_feature_x, "test_int_y": test_int_y, "test_onehot_y": test_onehot_y}

    return normalized_groups


## shuffle training set
def shuffleTrainingSet(normalized_groups):
    shuffled_groups = normalized_groups  # # use reference instead of copy to save memory
    for group_number, group_value in shuffled_groups.items():
        data_number = group_value['train_feature_x'].shape[-1]
        # Shuffles the indices
        idx = np.arange(data_number)
        np.random.shuffle(idx)
        train_idx = idx[: int(data_number)]
        # shuffle the data
        group_value['train_feature_x'], group_value['train_int_y'], group_value['train_onehot_y'], group_value['test_feature_x'] = (
        group_value['train_feature_x'][:, :, train_idx])[:, :, np.newaxis, :], group_value['train_int_y'][train_idx], group_value[
            'train_onehot_y'][train_idx, :], group_value['test_feature_x'][:, :, np.newaxis, :]

    return shuffled_groups

