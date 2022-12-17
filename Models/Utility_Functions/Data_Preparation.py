'''
load extracted emg features and create a cross validation dataset as well as group separation. it also includes the function for
producing the transfer learning dataset.
'''


## import modules
from Processing.Utility_Functions import Feature_Storage, Data_Reshaping
import random
import numpy as np
import copy


## load emg feature data
def loadEmgFeature(subject, version, feature_set):
    # load emg feature data
    emg_features = Feature_Storage.readEmgFeatures(subject, version, feature_set)
    # reorganize the data structure
    for gait_event_label, gait_event_features in emg_features.items():
        repetition_data = []
        for repetition_label, repetition_features in gait_event_features.items():
            repetition_data.append(np.array(repetition_features))  # convert 2d list into numpy
        emg_features[gait_event_label] = repetition_data  # convert repetition data from dict into list
    # if you want to use CNN model, you need to reshape the data
    emg_feature_reshaped = Data_Reshaping.reshapeEmgFeatures(emg_features)
    return emg_features, emg_feature_reshaped


## abandon samples from some modes
def removeSomeSamples(emg_features, start_index=0, end_index=-1):
    emg_feature_data = copy.deepcopy(emg_features)
    # remove some modes
    emg_feature_data['emg_LWLW_features'] = emg_feature_data['emg_LWLW_features'][
    int(len(emg_feature_data['emg_LWLW_features']) / 4): int(len(emg_feature_data['emg_LWLW_features']) * 3 / 4)]  # remove half of LWLW mode
    emg_feature_data.pop('emg_LW_features', None)
    emg_feature_data.pop('emg_SD_features', None)
    emg_feature_data.pop('emg_SA_features', None)
    # emg_feature_data.pop('emg_SSSS_features', None)

    # remove some feature data from each repetition
    if start_index != 0 and end_index != -1:  # no sample is removed if 'start_index != 0 and end_index != -1'
        if emg_feature_data['emg_LWSA_features'][0].ndim == 2:  # if emg data is 1d (2 dimension matrix)
            for transition_type, transition_features in emg_feature_data.items():
                for count, value in enumerate(transition_features):
                    transition_features[count] = value[start_index: end_index + 1, :]  # only keep the feature data between start index and end index
        elif emg_feature_data['emg_LWSA_features'][0].ndim == 4:  # if emg data is 2d (4 dimension matrix)
            for transition_type, transition_features in emg_feature_data.items():
                for count, value in enumerate(transition_features):
                    transition_features[count] = value[:, :, :, start_index: end_index + 1]  # only keep the feature data between start index and end index

    return emg_feature_data

## create k-fold cross validation groups
def crossValidationSet(fold, emg_feature_data):
    cross_validation_groups = {}  # 5 groups of cross validation set
    for i in range(fold):
        train_set = {}  # store train set of all gait events for each group
        test_set = {}  # store test set of all gait events for each group
        for gait_event_label, gait_event_features in copy.deepcopy(emg_feature_data).items():
            # shuffle the list (important)
            random.Random(5).shuffle(gait_event_features)  # 5 is a seed
            # separate the training and test set
            test_set[gait_event_label] = gait_event_features[
            int(len(gait_event_features) * i / fold): int(len(gait_event_features) * (i + 1) / fold)]
            del gait_event_features[  # remove test set from original set
            int(len(gait_event_features) * i / fold):  int(len(gait_event_features) * (i + 1) / fold)]
            train_set[gait_event_label] = gait_event_features
        cross_validation_groups[f"group_{i}"] = {"train_set": train_set, "test_set": test_set}  # a pair of training and test set for one group
    return cross_validation_groups


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
if __name__ == '__main__':
    #  class name to labels (according to the alphabetical order)
    class_all = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_LW_features': 4,
        'emg_SALW_features': 5, 'emg_SASA_features': 6, 'emg_SASS_features': 7, 'emg_SA_features': 8, 'emg_SDLW_features': 9,
        'emg_SDSD_features': 10, 'emg_SDSS_features': 11, 'emg_SD_features': 12, 'emg_SSLW_features': 13, 'emg_SSSA_features': 14,
        'emg_SSSD_features': 15, 'emg_SSSS_features': 16}
    class_reduced = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_SALW_features': 4,
        'emg_SASA_features': 5, 'emg_SASS_features': 6, 'emg_SDLW_features': 7, 'emg_SDSD_features': 8, 'emg_SDSS_features': 9,
        'emg_SSLW_features': 10, 'emg_SSSA_features': 11, 'emg_SSSD_features': 12, 'emg_SSSS_features': 13}