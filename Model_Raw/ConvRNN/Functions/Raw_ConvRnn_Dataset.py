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
def shuffleTrainingSet(normalized_groups, feature_window_per_repetition, predict_window_shift_unit, predict_using_window_number):
    shuffled_groups = copy.deepcopy(normalized_groups)

    # combine the data from all repetitions into one single list
    shift_range = feature_window_per_repetition - predict_using_window_number  # decide how many shifts there are in the for loop below
    for group_number, group_value in shuffled_groups.items():
        for key, value in group_value.items():
            if key == 'train_feature_x' or key == 'test_feature_x':
                for repetition_number, repetition_value in enumerate(value):
                    repetition_data = []
                    for i in range(0, shift_range + 1, predict_window_shift_unit):
                        repetition_data.append(repetition_value[:, :, i:i+predict_using_window_number][:, :, np.newaxis, :])
                    value[repetition_number] = repetition_data
                group_value[key] = [item for sublist in value for item in sublist]  # convert nested list to a single list
            elif key == 'train_int_y' or key == 'test_int_y' or key == 'train_onehot_y' or key == 'test_onehot_y':
                group_value[key] = np.repeat(value, len(range(0, shift_range + 1, predict_window_shift_unit)), axis=0)  # repeat y to match the number of x

        # Shuffles the indices
        data_number = len(group_value['train_feature_x'])
        idx = np.random.permutation(data_number)
        group_value['train_feature_x'] = [group_value['train_feature_x'][i] for i in idx]
        group_value['train_int_y'] = group_value['train_int_y'][idx]
        group_value['train_onehot_y'] = group_value['train_onehot_y'][idx]

    return shuffled_groups


# ##
# def combineNormalizedDataset(sliding_window_dataset):
#     normalized_groups = {}
#     for group_number, group_value in sliding_window_dataset.items():
#         # initialize training set and test set for each group
#         train_data, train_labels = [], []
#         test_data, test_labels = [], []
#
#         for set_type, set_data in group_value.items():
#             if set_type == 'train_set':
#                 train_data.extend(set_data.values())
#                 train_labels.extend([label for label in set_data.keys() for _ in range(len(set_data[label]))])
#             elif set_type == 'test_set':
#                 test_data.extend(set_data.values())
#                 test_labels.extend([label for label in set_data.keys() for _ in range(len(set_data[label]))])
#
#         train_feature_x, test_feature_x = standardize_data(train_data, test_data)
#         train_int_y, train_onehot_y, test_int_y, test_onehot_y = encode_labels(train_labels, test_labels)
#
#         normalized_groups[group_number] = {"train_feature_x": train_feature_x, "train_int_y": train_int_y, "train_onehot_y": train_onehot_y,
#             "test_feature_x": test_feature_x, "test_int_y": test_int_y, "test_onehot_y": test_onehot_y}
#
#     return normalized_groups
#
# def standardize_data(train_data, test_data):
#     # mean and std value for the training data
#     train_data_concat = np.concatenate(train_data, axis=-1)
#     mean = np.mean(train_data_concat, axis=-1)[:, :, np.newaxis]
#     std = np.std(train_data_concat, axis=-1)[:, :, np.newaxis]
#
#     # standardize the training and test data
#     train_data = [(d - mean) / std for d in train_data]
#     test_data = [(d - mean) / std for d in test_data]
#
#     return train_data, test_data
#
# def encode_labels(train_labels, test_labels):
#     # encode labels as integers and one-hot encode them
#     train_int_y = LabelEncoder().fit_transform(train_labels)
#     train_onehot_y = tf.keras.utils.to_categorical(train_int_y)
#     test_int_y = LabelEncoder().fit_transform(test_labels)
#     test_onehot_y = tf.keras.utils.to_categorical(test_int_y)
#
#     return train_int_y, train_onehot_y, test_int_y, test_onehot_y


# ##
# def shuffleTrainingSet(normalized_groups, predict_window_per_repetition, predict_window_shift_unit, predict_using_window_number):
#     shuffled_groups = normalized_groups  # use reference instead of copy to save memory
#
#     for group_value in shuffled_groups.values():
#         for key, value in group_value.items():
#             if key in ['train_feature_x', 'test_feature_x']:
#                 repetition_data = [
#                     np.moveaxis(np.concatenate([repetition_value[:, :, i:i + predict_using_window_number + 1][:, :, np.newaxis, :]], axis=-2),
#                         -2, 0) for repetition_value in value for i in range(0, predict_window_per_repetition, predict_window_shift_unit)]
#                 group_value[key] = repetition_data
#             elif key in ['train_int_y', 'test_int_y', 'train_onehot_y', 'test_onehot_y']:
#                 group_value[key] = np.repeat(value, predict_window_per_repetition, axis=0)
#
#         # Shuffle the training data
#         data_number = len(group_value['train_feature_x'])
#         idx = np.random.permutation(data_number)
#         group_value['train_feature_x'] = [group_value['train_feature_x'][i] for i in idx]
#         group_value['train_int_y'] = group_value['train_int_y'][idx]
#         group_value['train_onehot_y'] = group_value['train_onehot_y'][idx]
#
#     return shuffled_groups