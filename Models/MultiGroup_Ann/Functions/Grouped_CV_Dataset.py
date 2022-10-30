## import modules
import copy
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from Models.Basic_Ann.Functions import CV_Dataset   # for importing emg feature data


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
                    feature_x.extend(np.concatenate(gait_event_features, axis=0))
                    feature_y.extend([gait_event_label] * len(gait_event_features) * window_per_repetition)
                    combined_groups[group_number][set_type][transition_type].pop(
                        gait_event_label)  # abandon this value as it has been moved to a new variable
                # encode categories
                int_y = LabelEncoder().fit_transform(feature_y)  # int style categories
                onehot_y = tf.keras.utils.to_categorical(int_y)  # one-hot style categories (according to the alphabetical order)
                combined_groups[group_number][set_type][transition_type]['feature_x'] = np.array(feature_x)
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
                        transition_value['feature_x'], axis=0)) / np.std(transition_value['feature_x'], axis=0)
            elif set_type == 'test_set':
                for transition_type, transition_value in set_value.items():
                    train_x = normalized_groups[group_number]['train_set'][transition_type]['feature_x']
                    normalized_groups[group_number][set_type][transition_type]['feature_norm_x'] = (transition_value['feature_x'] - np.mean(
                        train_x, axis=0)) / np.std(train_x, axis=0)
    return normalized_groups


## shuffle training set
def shuffleTrainingSet(normalized_groups):
    shuffled_groups = copy.deepcopy(normalized_groups)
    for group_number, group_value in shuffled_groups.items():
        for set_type, set_value in group_value.items():
            if set_type == 'train_set':
                for transition_type, transition_value in set_value.items():
                    data_number = len(transition_value['feature_x'])
                    # Shuffles the indices
                    idx = np.arange(data_number)
                    np.random.shuffle(idx)
                    train_idx = idx[: int(data_number)]
                    # shuffle the data
                    transition_value['feature_x'], transition_value['feature_norm_x'], transition_value['feature_int_y'], transition_value[
                        'feature_onehot_y'] = transition_value['feature_x'][train_idx, :], transition_value['feature_norm_x'][train_idx, :], \
                        transition_value['feature_int_y'][train_idx], transition_value['feature_onehot_y'][train_idx, :]
    return shuffled_groups


##  obtain grouped k-fold cross-validation dataset
if __name__ == '__main__':
    # basic information
    subject = "Shibo"
    version = 1  # which experiment data to process
    feature_set = 1  # which feature set to use

    # read feature data
    emg_features, emg_feature_reshaped = CV_Dataset.loadEmgFeature(subject, version, feature_set)
    emg_feature_data = CV_Dataset.removeSomeMode(emg_features)
    window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[0]  # how many windows there are for each event repetition
    fold = 5  # 5-fold cross validation
    cross_validation_groups = CV_Dataset.crossValidationSet(fold, emg_feature_data)

    # reorganize data
    transition_grouped = separateGroups(cross_validation_groups)
    combined_groups = combineIntoDataset(transition_grouped, window_per_repetition)
    normalized_groups = normalizeDataset(combined_groups)
    shuffled_groups = shuffleTrainingSet(normalized_groups)

    #  class name to labels (according to the alphabetical order)
    class_all = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_LW_features': 4,
        'emg_SALW_features': 5, 'emg_SASA_features': 6, 'emg_SASS_features': 7, 'emg_SA_features': 8, 'emg_SDLW_features': 9,
        'emg_SDSD_features': 10, 'emg_SDSS_features': 11, 'emg_SD_features': 12, 'emg_SSLW_features': 13, 'emg_SSSA_features': 14,
        'emg_SSSD_features': 15}
    class_reduced = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_SALW_features': 4,
        'emg_SASA_features': 5, 'emg_SASS_features': 6, 'emg_SDLW_features': 7, 'emg_SDSD_features': 8, 'emg_SDSS_features': 9,
        'emg_SSLW_features': 10, 'emg_SSSA_features': 11, 'emg_SSSD_features': 12}