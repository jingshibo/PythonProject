'''
load extracted emg features and create a cross validation dataset as well as group separation. it also includes the function for
producing the transfer learning dataset.
'''


## import modules
from Pre_Processing.Utility_Functions import Feature_Storage, Data_Reshaping
import random
import numpy as np
import copy
import math


## load emg feature data
def loadEmgFeature(subject, version, feature_set):
    # load emg feature data
    emg_features = Feature_Storage.readEmgFeatures(subject, version, feature_set)
    # reorganize the data structure
    for gait_event_label, gait_event_features in emg_features.items():
        repetition_data = []
        for repetition_label, repetition_features in gait_event_features.items():
            # if repetition_features:  # if list is not empty
            repetition_data.append(np.array(repetition_features).astype(np.float32))  # convert 2d list into numpy
        emg_features[gait_event_label] = repetition_data  # convert repetition data from dict into list
    # if you want to use CNN model, you need to reshape the data
    emg_feature_reshaped = Data_Reshaping.reshapeEmgFeatures(emg_features)
    return emg_features, emg_feature_reshaped


## abandon samples from some modes
def removeSomeSamples(emg_all_data, start_index=0, end_index=-1, is_down_sampling=False):
    emg_selected_data = copy.deepcopy(emg_all_data)

    # check if emg_features is raw data or feature data
    if 'emg_LWLW_features' in list(emg_all_data.keys()):  # emg features
        string = '_features'
    elif 'emg_LWLW' in list(emg_all_data.keys()):  # emg data
        string = ''

    # remove some modes
    emg_selected_data[f'emg_LWLW{string}'] = emg_selected_data[f'emg_LWLW{string}'][
    int(len(emg_selected_data[f'emg_LWLW{string}']) / 4): int(len(emg_selected_data[f'emg_LWLW{string}']) * 3 / 4)]  # remove half of LWLW mode
    emg_selected_data.pop(f'emg_LW{string}', None)
    emg_selected_data.pop(f'emg_SD{string}', None)
    emg_selected_data.pop(f'emg_SA{string}', None)
    emg_selected_data.pop(f'emg_SSSS{string}', None)

    # remove some feature data from each repetition
    if end_index != -1:  # no sample is removed if end_index != -1'
        if emg_selected_data[f'emg_LWSA{string}'][0].ndim == 2:  # if emg data is 1d (2 dimension matrix)
            for transition_type, transition_data in emg_selected_data.items():
                for count, value in enumerate(transition_data):
                    transition_data[count] = value[start_index: end_index + 1, :]  # only keep the feature data between start index and end index
        elif emg_selected_data[f'emg_LWSA{string}'][0].ndim == 4:  # if emg data is 2d (4 dimension matrix)
            for transition_type, transition_data in emg_selected_data.items():
                for count, value in enumerate(transition_data):
                    transition_data[count] = value[:, :, :, start_index: end_index + 1]  # only keep the feature data between start index and end index

    if is_down_sampling == True:  # use a factor of 2 to downsample the emg data
        for transition_label, transition_data in emg_selected_data.items():
            for number, value in enumerate(transition_data):
                emg_selected_data[transition_label][number] = value[::2, :]  # [1 3 5 7 9]

    return emg_selected_data


## create k-fold cross validation groups
def crossValidationSet(fold, emg_feature_data, shuffle=True):
    cross_validation_groups = {}  # 5 groups of cross validation set
    for i in range(fold):
        train_set = {}  # store train set of all gait events for each group
        test_set = {}  # store test set of all gait events for each group
        for gait_event_label, gait_event_features in copy.deepcopy(emg_feature_data).items():
            # shuffle the list (important)
            if shuffle:
                random.Random(5).shuffle(gait_event_features)  # 5 is a seed
            else:
                pass
            # separate the training and test set
            test_set[gait_event_label] = gait_event_features[
            int(len(gait_event_features) * i / fold): int(len(gait_event_features) * (i + 1) / fold)]
            del gait_event_features[  # remove test set from original set
            int(len(gait_event_features) * i / fold):  int(len(gait_event_features) * (i + 1) / fold)]
            train_set[gait_event_label] = gait_event_features
        cross_validation_groups[f"group_{i}"] = {"train_set": train_set, "test_set": test_set}  # a pair of training and test set for one group
    return cross_validation_groups


## leaving the given percent of data as test set
def leaveOneSet(leave_percent, emg_feature_data, shuffle=True):
    leave_one_groups = {}  # 5 groups of cross validation set
    train_set = {}  # store train set of all gait events for each group
    test_set = {}  # store test set of all gait events for each group
    if leave_percent <= 0 or leave_percent >= 1:
        raise TypeError("The percentage should be within (0,1)")
    for gait_event_label, gait_event_features in copy.deepcopy(emg_feature_data).items():
        # shuffle the list (important)
        if shuffle:
            random.Random(5).shuffle(gait_event_features)  # 5 is a seed
        else:
            pass
        # separate the training and test set
        test_set[gait_event_label] = gait_event_features[math.ceil(len(gait_event_features) * (1 - leave_percent)):]
        del gait_event_features[math.ceil(len(gait_event_features) * (1 - leave_percent)):]  # remove test set from original set
        train_set[gait_event_label] = gait_event_features
    leave_one_groups[f"group_0"] = {"train_set": train_set, "test_set": test_set}  # a pair of training and test set for one group
    return leave_one_groups


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
                ["SSLW", "SSSA", "SSSD", "SSSS"]]  # 4 groups
            # transition_groups = [["LWLW", "LWSA", "LWSD", "LWSS"], ["SALW", "SASA", "SASS"], ["SDLW", "SDSD", "SDSS"],
            #     ["SSLW", "SSSA", "SSSD"]]  # 4 groups
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


## divide the dataset specifically for transfer learning
def divideTransferDataset(fold, emg_feature_data, transfer_data_percent):
    pre_train_groups = {}  # 5 groups of cross validation set
    transfer_train_groups = {}  # 5 groups of cross validation set
    for i in range(fold):
        train_set = {}  # store train set of all gait events for each group
        test_set = {}  # store test set of all gait events for each group
        for gait_event_label, gait_event_features in copy.deepcopy(emg_feature_data).items():
            # shuffle the list (important)
            random.Random(5).shuffle(gait_event_features)  # 4 is a seed
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
            random.Random(5).shuffle(gait_event_features)  # 4 is a seed
            # separate the pre_train and transfer_train set
            transfer_train_set[gait_event_label] = gait_event_features[0: int(len(gait_event_features) * transfer_data_percent)]
            del gait_event_features[0:  int(len(gait_event_features) * transfer_data_percent)]  # remove transfer_train set from train set
            pre_train_set[gait_event_label] = gait_event_features
        pre_train_groups[f"group_{i}"] = {"train_set": pre_train_set, "test_set": test_set}  # a pair of pretrain and test set for one group
        transfer_train_groups[f"group_{i}"] = {"train_set": transfer_train_set, "test_set": test_set}  # a pair of transfer train and test set for one group
    return pre_train_groups,  transfer_train_groups


##
class_all = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_LW_features': 4,
    'emg_SALW_features': 5, 'emg_SASA_features': 6, 'emg_SASS_features': 7, 'emg_SA_features': 8, 'emg_SDLW_features': 9,
    'emg_SDSD_features': 10, 'emg_SDSS_features': 11, 'emg_SD_features': 12, 'emg_SSLW_features': 13, 'emg_SSSA_features': 14,
    'emg_SSSD_features': 15, 'emg_SSSS_features': 16}
class_reduced = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_SALW_features': 4,
    'emg_SASA_features': 5, 'emg_SASS_features': 6, 'emg_SDLW_features': 7, 'emg_SDSD_features': 8, 'emg_SDSS_features': 9,
    'emg_SSLW_features': 10, 'emg_SSSA_features': 11, 'emg_SSSD_features': 12, 'emg_SSSS_features': 13}