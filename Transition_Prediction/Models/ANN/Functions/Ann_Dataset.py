'''
create a dataset for a single ann model with normalization and shuffling
'''

## import modules
import copy
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


## combine data of all gait events into a single dataset
def combineNormalizedDataset(cross_validation_groups, window_per_repetition):
    normalized_groups = {}

    for group_number, group_value in cross_validation_groups.items():
        # initialize training set and test set for each group
        train_feature_x = []
        train_feature_y = []
        test_feature_x = []
        test_feature_y = []
        for set_type, set_value in group_value.items():
            if set_type == 'train_set':  # combine all data into a dataset
                for gait_event_label, gait_event_features in set_value.items():
                    train_feature_x.extend(np.concatenate(gait_event_features))
                    train_feature_y.extend([gait_event_label] * len(gait_event_features) * window_per_repetition)
            elif set_type == 'test_set':  # keep the structure unchanged
                for gait_event_label, gait_event_features in set_value.items():
                    test_feature_x.extend(np.concatenate(gait_event_features))
                    test_feature_y.extend([gait_event_label] * len(gait_event_features) * window_per_repetition)
        # normalization
        train_norm_x = (train_feature_x - np.mean(train_feature_x, axis=0)) / np.std(train_feature_x, axis=0)
        test_norm_x = (test_feature_x - np.mean(train_feature_x, axis=0)) / np.std(train_feature_x, axis=0)
        # one-hot encode categories (according to the alphabetical order)
        train_int_y = LabelEncoder().fit_transform(train_feature_y)
        train_onehot_y = tf.keras.utils.to_categorical(train_int_y)
        test_int_y = LabelEncoder().fit_transform(test_feature_y)
        test_onehot_y = tf.keras.utils.to_categorical(test_int_y)
        # put training data and test data into one group
        normalized_groups[group_number] = {"train_feature_x": train_norm_x, "train_int_y": train_int_y, "train_onehot_y": train_onehot_y,
            "test_feature_x": test_norm_x, "test_int_y": test_int_y, "test_onehot_y": test_onehot_y}
    return normalized_groups


## shuffle training set
def shuffleTrainingSet(normalized_groups):
    shuffled_groups = copy.deepcopy(normalized_groups)
    for group_number, group_value in shuffled_groups.items():
        data_number = len(group_value['train_feature_x'])
        # Shuffles the indices
        idx = np.arange(data_number)
        np.random.shuffle(idx)
        train_idx = idx[: int(data_number)]
        # shuffle the data
        group_value['train_feature_x'], group_value['train_int_y'], group_value['train_onehot_y'] = group_value['train_feature_x'][
        train_idx, :], group_value['train_int_y'][train_idx], group_value['train_onehot_y'][train_idx, :]
    return shuffled_groups


## only use emg data before gait events
# emg_x = emg_feature_x[0::17, :]
# emg_y = emg_feature_y[0::17]
# emg_feature_x = emg_x

##  only use selected emg channels
# emg_feature_tibialis_x = emg_feature_x[:, 0: 65]
# for i in range(1, 8):
#     emg_feature_tibialis_x = np.concatenate((emg_feature_tibialis_x, emg_feature_x[:, 0+130*i: 65+130*i]), axis=1)
# emg_feature_rectus_x = emg_feature_x[:, 65: 130]
# for i in range(1, 8):
#     emg_feature_rectus_x = np.concatenate((emg_feature_rectus_x, emg_feature_x[:, 65+130*i: 130+130*i]), axis=1)
# emg_feature_bipolar_x = emg_feature_x[:, 33].reshape(len(emg_feature_y), 1)
# for i in range(1, 16):
#     emg_feature_bipolar_x = np.concatenate((emg_feature_bipolar_x, emg_feature_x[:, 33+65*i].reshape(len(emg_feature_y), 1)), axis=1)
# emg_feature_x = emg_feature_tibialis_x


