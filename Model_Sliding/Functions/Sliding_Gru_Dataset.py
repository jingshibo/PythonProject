'''
create a dataset for a sliding gru model with normalization and shuffling
'''

##
import numpy as np
import copy
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


## select samples from a certain window position
def selectSamples(emg_features, start_index, end_index):
    emg_feature_data = copy.deepcopy(emg_features)
    # select some feature data from each repetition
    if emg_feature_data['emg_LWSA_features'][0].ndim == 2:  # if emg data is 1d (2 dimension matrix)
        for transition_type, transition_features in emg_feature_data.items():
            for count, value in enumerate(transition_features):
                transition_features[count] = value[start_index: end_index + 1, :]  # only keep the feature data between start index and end index
    elif emg_feature_data['emg_LWSA_features'][0].ndim == 4:  # if emg data is 2d (4 dimension matrix)
        for transition_type, transition_features in emg_feature_data.items():
            for count, value in enumerate(transition_features):
                transition_features[count] = value[:, :, :, start_index: end_index + 1]  # only keep the feature data between start index and end index

    return emg_feature_data


##  create emg dataset from different sliding window positions
def createSlidingDataset(cross_validation_groups, shift_unit, initial_start=0, initial_end=16):
    # initial_start is the start window position, initial_end is the end window position, shift_unit defines the number of window shift for each sliding
    window_per_repetition = cross_validation_groups['group_0']['train_set']['emg_LWLW_features'][0].shape[0]  # how many windows for each event repetition
    shift_range = window_per_repetition - initial_start - initial_end  # decide how many shifts to do in the for loop below

    emg_feature_all = copy.deepcopy(cross_validation_groups)
    for group_number, group_value in emg_feature_all.items():
        for set_type, set_value in group_value.items():
            emg_sliding_features = {}  # emg features from different window positions
            if set_type == 'train_set':  # combine the features from different window positions together
                for shift in range(0, shift_range, shift_unit):  # shift = 0, 4, 8, 12, 16, 20, 24, 28, 32
                    emg_feature_value = selectSamples(set_value, start_index=initial_start + shift, end_index=initial_end + shift)  # (0,16) -> (32,48)
                    for gait_event_label, gait_event_emg in emg_feature_value.items():
                        if gait_event_label in emg_sliding_features:  # check if there is already the key in the dict
                            emg_sliding_features[gait_event_label].extend(gait_event_emg)
                        else:
                            emg_sliding_features[gait_event_label] = gait_event_emg
            elif set_type == 'test_set':  # put the features from different window positions separately
                for shift in range(0, shift_range, shift_unit):  # 0, 4, 8, 12, 16, 20, 24, 28, 32
                    emg_feature_value = selectSamples(set_value, start_index=initial_start + shift, end_index=initial_end + shift)
                    emg_sliding_features[f'shift_{shift}'] = emg_feature_value  # the keyword is the number of window shift
            group_value[set_type] = emg_sliding_features

    return emg_feature_all


## combine data of all gait events into a single dataset
def combineNormalizedDataset(emg_sliding_features, window_per_repetition):
    normalized_groups = {}
    # window_per_repetition = emg_feature_all['group_0']['train_set']['emg_LWLW_features'][0].shape[0]  # how many windows for each event repetition

    for group_number, group_value in emg_sliding_features.items():
        # initialize training set for each group
        train_feature_x = []
        train_feature_y = []
        for gait_event_label, gait_event_features in group_value['train_set'].items():  # training set: combine all data into a dataset
            train_feature_x.extend(np.concatenate(gait_event_features))
            train_feature_y.extend([gait_event_label] * len(gait_event_features) * window_per_repetition)
        # normalization
        train_norm_x = (train_feature_x - np.mean(train_feature_x, axis=0)) / np.std(train_feature_x, axis=0)
        # one-hot encode categories (according to the alphabetical order)
        train_int_y = LabelEncoder().fit_transform(train_feature_y)
        train_onehot_y = tf.keras.utils.to_categorical(train_int_y)
        #  reshape the data structure for RNN model (of shape [batch, timesteps, feature])
        train_norm_x_reshaped = np.transpose(np.reshape(train_norm_x, (window_per_repetition, -1, train_norm_x.shape[1]), order='F'), (1, 0, 2))
        train_int_y_reshaped = np.transpose(np.reshape(train_int_y, (window_per_repetition, -1), order='F'))
        train_onehot_y_reshaped = np.transpose(np.reshape(train_onehot_y, (window_per_repetition, -1, train_onehot_y.shape[1]), order='F'), (1, 0, 2))

        train_set = {"train_feature_x": train_norm_x_reshaped, "train_int_y": train_int_y_reshaped, "train_onehot_y": train_onehot_y_reshaped}

        test_set = {}
        for shift_number, shift_value in group_value['test_set'].items():  # test set: keep the structure unchanged
            # initialize test set for each shift
            test_feature_x = []
            test_feature_y = []
            for gait_event_label, gait_event_features in shift_value.items():
                test_feature_x.extend(np.concatenate(gait_event_features))
                test_feature_y.extend([gait_event_label] * len(gait_event_features) * window_per_repetition)
            # normalization
            test_norm_x = (test_feature_x - np.mean(train_feature_x, axis=0)) / np.std(train_feature_x, axis=0)
            # one-hot encode categories (according to the alphabetical order)
            test_int_y = LabelEncoder().fit_transform(test_feature_y)
            test_onehot_y = tf.keras.utils.to_categorical(test_int_y)
            #  reshape the data structure for RNN model (of shape [batch, timesteps, feature])
            test_norm_x_reshaped = np.transpose(np.reshape(test_norm_x, (window_per_repetition, -1, test_norm_x.shape[1]), order='F'), (1, 0, 2))
            test_int_y_reshaped = np.transpose(np.reshape(test_int_y, (window_per_repetition, -1), order='F'))
            test_onehot_y_reshaped = np.transpose(np.reshape(test_onehot_y, (window_per_repetition, -1, test_onehot_y.shape[1]), order='F'), (1, 0, 2))
            test_set[shift_number] = {"test_feature_x": test_norm_x_reshaped, "test_int_y": test_int_y_reshaped, "test_onehot_y": test_onehot_y_reshaped}

        # put training data and test data into one group
        normalized_groups[group_number] = {"train_set": train_set, "test_set": test_set}
    return normalized_groups


## shuffle training set
def shuffleTrainingSet(normalized_groups):
    shuffled_groups = copy.deepcopy(normalized_groups)
    for group_number, group_value in shuffled_groups.items():
        data_number = group_value['train_set']['train_feature_x'].shape[0]
        # Shuffles the indices
        idx = np.arange(data_number)
        np.random.shuffle(idx)
        train_idx = idx[: int(data_number)]
        # shuffle the data
        group_value['train_set']['train_feature_x'], group_value['train_set']['train_int_y'], group_value['train_set']['train_onehot_y'] = \
        group_value['train_set']['train_feature_x'][train_idx, :, :], group_value['train_set']['train_int_y'][train_idx, :], \
        group_value['train_set']['train_onehot_y'][train_idx, :, :]
    return shuffled_groups


## select specific channels for model training and testing
def select1dFeatureChannels(group_value, channel_to_calculate):
    # training dataset
    train_data = group_value['train_set']
    train_emg1_x = train_data['train_feature_x'][:, :, 0: 65]
    train_emg2_x = train_data['train_feature_x'][:, :, 65: 130]
    num_features = int(group_value['train_set']['train_feature_x'].shape[-1] / 130)  # which features to compute

    for i in range(1, num_features):  # extract emg1 data and emg2 data
        train_emg1_x = np.concatenate((train_emg1_x, train_data['train_feature_x'][:, :, 0 + 130 * i: 65 + 130 * i]), axis=-1)
        train_emg2_x = np.concatenate((train_emg2_x, train_data['train_feature_x'][:, :, 65 + 130 * i: 130 + 130 * i]), axis=-1)
    train_set_y = train_data['train_onehot_y'][:, 0, :]
    train_int_y = train_data['train_int_y'][:, 0]

    if channel_to_calculate == 'emg_1':
        train_set_x = train_emg1_x[:, :, 0: 65 * num_features]
    elif channel_to_calculate == 'emg_2':
        train_set_x = train_emg2_x[:, :, 0: 65 * num_features]
    elif channel_to_calculate == 'emg_all':
        train_set_x = train_data['train_feature_x'][:, :, 0: 130 * num_features]
    elif channel_to_calculate == 'emg_bipolar':
        pass
        # bipolar input data
        # train_bipolar_x = group_value['train_feature_x'][:, 33].reshape(len(group_value['train_int_y']), 1)
        # for i in range(1, 16):
        #     emg_feature_bipolar_x = np.concatenate((train_bipolar_x, group_value[:, 33+65*i].reshape(len(group_value['train_int_y']), 1)), axis=1)
    else:
        raise Exception("No Such Channels")
    train_set = {'train_set_x': train_set_x, 'train_set_y': train_set_y, 'train_int_y': train_int_y}

    # test dataset
    test_set = {}
    test_data = group_value['test_set']
    for shift_number, shift_value in test_data.items():
        test_emg1_x = shift_value['test_feature_x'][:, :, 0: 65]
        test_emg2_x = shift_value['test_feature_x'][:, :, 65: 130]
        for i in range(1, num_features):  # extract emg1 data and emg2 data
            test_emg1_x = np.concatenate((test_emg1_x, shift_value['test_feature_x'][:, :, 0 + 130 * i: 65 + 130 * i]), axis=-1)
            test_emg2_x = np.concatenate((test_emg2_x, shift_value['test_feature_x'][:, :, 65 + 130 * i: 130 + 130 * i]), axis=-1)
        test_set_y = shift_value['test_onehot_y'][:, 0, :]
        test_int_y = shift_value['test_int_y'][:, 0]

        if channel_to_calculate == 'emg_1':
            test_set_x = test_emg1_x[:, :, 0: 65 * num_features]
        elif channel_to_calculate == 'emg_2':
            test_set_x = test_emg2_x[:, :, 0: 65 * num_features]
        elif channel_to_calculate == 'emg_all':
            test_set_x = shift_value['test_feature_x'][:, :, 0: 130 * num_features]
        elif channel_to_calculate == 'emg_bipolar':
            pass
            # bipolar input data
            # train_bipolar_x = group_value['train_feature_x'][:, 33].reshape(len(group_value['train_int_y']), 1)
            # for i in range(1, 16):
            #     emg_feature_bipolar_x = np.concatenate((train_bipolar_x, group_value[:, 33+65*i].reshape(len(group_value['train_int_y']), 1)), axis=1)
        else:
            raise Exception("No Such Channels")
        test_set[shift_number] = {'test_set_x': test_set_x, 'test_set_y': test_set_y, 'test_int_y': test_int_y}

    return train_set, test_set

