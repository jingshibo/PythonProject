## import modules
import copy
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from Processing.Utility_Functions import Feature_Storage, Data_Reshaping


## load emg feature data
def loadEmgFeature(subject, version, feature_set):
    # load emg feature data
    emg_features = Feature_Storage.readEmgFeatures(subject, version, feature_set)
    # reorganize the data structure
    for gait_event_label, gait_event_features in emg_features.items():
        repetition_data = []
        for repetition_label, repetition_features in gait_event_features.items():
            repetition_data.append(np.array(repetition_features))  # convert 2d list into numpy
        emg_features[gait_event_label] = repetition_data  # move repetition data from dict into list
    # if you want to use CNN model, you need to reshape the data
    emg_feature_reshaped = Data_Reshaping.reshapeEmgFeatures(emg_features)
    return emg_features, emg_feature_reshaped


## abandon samples from some modes
def removeSomeMode(emg_features):
    emg_feature_data = copy.deepcopy(emg_features)
    # emg_feature_data = emg_features
    emg_feature_data['emg_LWLW_features'] = emg_feature_data['emg_LWLW_features'][
    int(len(emg_feature_data['emg_LWLW_features']) / 4): int(len(emg_feature_data['emg_LWLW_features']) * 3 / 4)]  # remove half LWLW mode
    emg_feature_data.pop('emg_LW_features', None)
    emg_feature_data.pop('emg_SD_features', None)
    emg_feature_data.pop('emg_SA_features', None)
    return emg_feature_data


## create k-fold cross validation groups
def crossValidationSet(fold, emg_feature_data):
    cross_validation_groups = {}  # 5 groups of cross validation set
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
        cross_validation_groups[f"group_{i}"] = {"train_set": train_set, "test_set": test_set}  # a pair of training and test set for one group
    return cross_validation_groups


## combine data of all gait events into a single dataset
def combineIntoDataset(cross_validation_groups, window_per_repetition):
    normalized_groups = {}
    for group_number, group_value in cross_validation_groups.items():
        # initialize training set and test set for each group
        train_feature_x = []
        train_feature_y = []
        test_feature_x = []
        test_feature_y = []
        for set_type, set_value in group_value.items():
            for gait_event_label, gait_event_features in set_value.items():
                if set_type == 'train_set':  # combine all data into a dataset
                    train_feature_x.extend(np.concatenate(gait_event_features))
                    train_feature_y.extend([gait_event_label] * len(gait_event_features) * window_per_repetition)
                elif set_type == 'test_set':  # keep the structure unchanged
                    test_feature_x.extend(np.concatenate(gait_event_features))
                    test_feature_y.extend([gait_event_label] * len(gait_event_features) * window_per_repetition)
        # convert x from list to numpy
        train_feature_x = np.array(train_feature_x)
        test_feature_x = np.array(test_feature_x)
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


##  obtain k-fold cross-validation dataset
if __name__ == '__main__':
    # basic information
    subject = "Shibo"
    version = 1  # which experiment data to process
    feature_set = 1  # which feature set to use

    # read feature data
    emg_features, emg_feature_reshaped = loadEmgFeature(subject, version, feature_set)
    emg_feature_data = removeSomeMode(emg_features)
    window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[0]  # how many windows there are for each event repetition

    # reorganize data
    fold = 5  # 5-fold cross validation
    cross_validation_groups = crossValidationSet(fold, emg_feature_data)
    normalized_groups = combineIntoDataset(cross_validation_groups, window_per_repetition)
    shuffled_groups = shuffleTrainingSet(normalized_groups)

    #  class name to labels (according to the alphabetical order)
    class_all = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_LW_features': 4,
        'emg_SALW_features': 5, 'emg_SASA_features': 6, 'emg_SASS_features': 7, 'emg_SA_features': 8, 'emg_SDLW_features': 9,
        'emg_SDSD_features': 10, 'emg_SDSS_features': 11, 'emg_SD_features': 12, 'emg_SSLW_features': 13, 'emg_SSSA_features': 14,
        'emg_SSSD_features': 15}
    class_reduced = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_SALW_features': 4,
        'emg_SASA_features': 5, 'emg_SASS_features': 6, 'emg_SDLW_features': 7, 'emg_SDSD_features': 8, 'emg_SDSS_features': 9,
        'emg_SSLW_features': 10, 'emg_SSSA_features': 11, 'emg_SSSD_features': 12}


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


