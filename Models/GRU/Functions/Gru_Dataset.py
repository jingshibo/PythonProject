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

        #  reshape the data structure for RNN model (of shape [batch, timesteps, feature])
        train_norm_x_reshaped = np.transpose(np.reshape(train_norm_x, (window_per_repetition, -1, train_norm_x.shape[1]), order='F'), (1, 0, 2))
        test_norm_x_reshaped = np.transpose(np.reshape(test_norm_x, (window_per_repetition, -1, test_norm_x.shape[1]), order='F'), (1, 0, 2))
        train_int_y_reshaped = np.transpose(np.reshape(train_int_y, (window_per_repetition, -1), order='F'))
        test_int_y_reshaped = np.transpose(np.reshape(test_int_y, (window_per_repetition, -1), order='F'))
        train_onehot_y_reshaped = np.transpose(np.reshape(train_onehot_y, (window_per_repetition, -1, train_onehot_y.shape[1]), order='F'), (1, 0, 2))
        test_onehot_y_reshaped = np.transpose(np.reshape(test_onehot_y, (window_per_repetition, -1, test_onehot_y.shape[1]), order='F'), (1, 0, 2))

        # put training data and test data into one group
        normalized_groups[group_number] = {"train_feature_x": train_norm_x_reshaped, "train_int_y": train_int_y_reshaped, "train_onehot_y":
            train_onehot_y_reshaped, "test_feature_x": test_norm_x_reshaped, "test_int_y": test_int_y_reshaped, "test_onehot_y": test_onehot_y_reshaped}
    return normalized_groups


## shuffle training set
def shuffleTrainingSet(normalized_groups):
    shuffled_groups = copy.deepcopy(normalized_groups)
    for group_number, group_value in shuffled_groups.items():
        data_number = group_value['train_feature_x'].shape[0]
        # Shuffles the indices
        idx = np.arange(data_number)
        np.random.shuffle(idx)
        train_idx = idx[: int(data_number)]
        # shuffle the data
        group_value['train_feature_x'], group_value['train_int_y'], group_value['train_onehot_y'] = group_value['train_feature_x'][
        train_idx, :, :], group_value['train_int_y'][train_idx, :], group_value['train_onehot_y'][train_idx, :, :]
    return shuffled_groups


##
if __name__ == '__main__':
    #  class name to labels (according to the alphabetical order)
    class_all = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_LW_features': 4,
        'emg_SALW_features': 5, 'emg_SASA_features': 6, 'emg_SASS_features': 7, 'emg_SA_features': 8, 'emg_SDLW_features': 9,
        'emg_SDSD_features': 10, 'emg_SDSS_features': 11, 'emg_SD_features': 12, 'emg_SSLW_features': 13, 'emg_SSSA_features': 14,
        'emg_SSSD_features': 15}
    class_reduced = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_SALW_features': 4,
        'emg_SASA_features': 5, 'emg_SASS_features': 6, 'emg_SDLW_features': 7, 'emg_SDSD_features': 8, 'emg_SDSS_features': 9,
        'emg_SSLW_features': 10, 'emg_SSSA_features': 11, 'emg_SSSD_features': 12}
