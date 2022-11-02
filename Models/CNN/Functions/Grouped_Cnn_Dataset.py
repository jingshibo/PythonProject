## import modules
import copy
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from Models.Basic_Ann.Functions import Ann_Dataset   # for importing emg feature data
import random


## divide the dataset specifically for transfer learning
def divideTransferDataset(fold, emg_feature_data, transfer_data_percent):
    pre_train_groups = {}  # 5 groups of cross validation set
    transfer_train_groups = {}  # 5 groups of cross validation set
    for i in range(fold):
        train_set = {}  # store train set of all gait events for each group
        test_set = {}  # store test set of all gait events for each group
        for gait_event_label, gait_event_features in copy.deepcopy(emg_feature_data).items():
            # shuffle the list (important)
            random.Random(4).shuffle(gait_event_features)  # 4 is a seed
            # separate the training and test set
            test_set[gait_event_label] = gait_event_features[
            int(len(gait_event_features) * i / fold): int(len(gait_event_features) * (i + 1) / fold)]
            del gait_event_features[  # remove test set from original set
            int(len(gait_event_features) * i / fold):  int(len(gait_event_features) * (i + 1) / fold)]
            train_set[gait_event_label] = gait_event_features
        pre_train_set = {}
        transfer_train_set = {}
        for gait_event_label, gait_event_features in train_set.items():
            # shuffle the list
            random.Random(4).shuffle(gait_event_features)  # 4 is a seed
            # separate the pre_train and transfer_train set
            transfer_train_set[gait_event_label] = gait_event_features[0: int(len(gait_event_features) * transfer_data_percent)]
            del gait_event_features[0:  int(len(gait_event_features) * transfer_data_percent)]  # remove transfer_train set from train set
            pre_train_set[gait_event_label] = gait_event_features
        pre_train_groups[f"group_{i}"] = {"train_set": pre_train_set, "test_set": test_set}  # a pair of pretrain and test set for one group
        transfer_train_groups[f"group_{i}"] = {"train_set": transfer_train_set, "test_set": test_set}  # a pair of transfer train and test set for one group
    return pre_train_groups,  transfer_train_groups


## divide into 4 transition groups
def separateGroups(cross_validation_groups):
    transition_grouped = copy.deepcopy(cross_validation_groups)
    for group_number, group_value in cross_validation_groups.items():
        for set_type, set_value in group_value.items():
            transition_LW = {}
            transition_SA = {}
            transition_SD = {}
            transition_SS = {}
            transition_groups = [["LWLW", "LWSA", "LWSD", "LWSS"], ["SALW", "SASA", "SASS"], ["SDLW", "SDSD", "SDSS"],
                ["SSLW", "SSSA", "SSSD"]]  # 4 groups
            for transition_type, transition_value in set_value.items():
                if any(x in transition_type for x in transition_groups[0]):
                    transition_LW[transition_type] = transition_value
                elif any(x in transition_type for x in transition_groups[1]):
                    transition_SA[transition_type] = transition_value
                elif any(x in transition_type for x in transition_groups[2]):
                    transition_SD[transition_type] = transition_value
                elif any(x in transition_type for x in transition_groups[3]):
                    transition_SS[transition_type] = transition_value
                transition_grouped[group_number][set_type].pop(transition_type)  # abandon this value as it has been moved to the group
            transition_grouped[group_number][set_type]["transition_LW"] = transition_LW
            transition_grouped[group_number][set_type]["transition_SA"] = transition_SA
            transition_grouped[group_number][set_type]["transition_SD"] = transition_SD
            transition_grouped[group_number][set_type]["transition_SS"] = transition_SS
    return transition_grouped


## combine data of all gait events into a single dataset
def combineIntoDataset(transition_grouped, window_per_repetition):
    combined_groups = copy.deepcopy(transition_grouped)
    for group_number, group_value in transition_grouped.items():
        for set_type, set_value in group_value.items():
            for transition_type, transition_data in set_value.items():
                # combine the data within a transition group into one dataset
                feature_x = []
                feature_y = []
                for gait_event_label, gait_event_features in transition_data.items():
                    feature_x.append(np.concatenate(gait_event_features, axis=-1))
                    feature_y.extend([gait_event_label] * len(gait_event_features) * window_per_repetition)
                    combined_groups[group_number][set_type][transition_type].pop(
                        gait_event_label)  # abandon this value as it has been moved to a new variable
                feature_x = np.concatenate(feature_x, axis=-1)
                # encode categories
                int_y = LabelEncoder().fit_transform(feature_y)  # int style categories
                onehot_y = tf.keras.utils.to_categorical(int_y)  # one-hot style categories (according to the alphabetical order)
                combined_groups[group_number][set_type][transition_type]['feature_x'] = feature_x
                combined_groups[group_number][set_type][transition_type]['feature_int_y'] = int_y
                combined_groups[group_number][set_type][transition_type]['feature_onehot_y'] = onehot_y
    return combined_groups


## normalize dataset
def normalizeDataset(combined_groups):
    normalized_groups = copy.deepcopy(combined_groups)
    for group_number, group_value in combined_groups.items():
        for set_type, set_value in group_value.items():
            if set_type == 'train_set':
                for transition_type, transition_value in set_value.items():
                    normalized_groups[group_number][set_type][transition_type]['feature_norm_x'] = (transition_value['feature_x'] - np.mean(
                        transition_value['feature_x'], axis=-1)[:, :, :, np.newaxis]) / np.std(transition_value['feature_x'], axis=-1)[:, :, :, np.newaxis]
            elif set_type == 'test_set':
                for transition_type, transition_value in set_value.items():
                    train_x = normalized_groups[group_number]['train_set'][transition_type]['feature_x']
                    normalized_groups[group_number][set_type][transition_type]['feature_norm_x'] = (transition_value['feature_x'] - np.mean(
                        train_x, axis=-1)[:, :, :, np.newaxis]) / np.std(train_x, axis=-1)[:, :, :, np.newaxis]
    return normalized_groups


## shuffle training set
def shuffleTrainingSet(normalized_groups):
    shuffled_groups = copy.deepcopy(normalized_groups)
    for group_number, group_value in shuffled_groups.items():
        for set_type, set_value in group_value.items():
            if set_type == 'train_set':
                for transition_type, transition_value in set_value.items():
                    data_number = transition_value['feature_x'].shape[-1]
                    # Shuffles the indices
                    idx = np.arange(data_number)
                    np.random.shuffle(idx)
                    train_idx = idx[: int(data_number)]
                    # shuffle the data
                    transition_value['feature_x'], transition_value['feature_norm_x'], transition_value['feature_int_y'], transition_value[
                        'feature_onehot_y'] = transition_value['feature_x'][:, :, :, train_idx], transition_value['feature_norm_x'][:, :, :,
                    train_idx], transition_value['feature_int_y'][train_idx], transition_value['feature_onehot_y'][train_idx, :]
    return shuffled_groups
