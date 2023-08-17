'''
create multiple cnn dataset for different transition groups and then normalize, shuffle the dataset within the group.
'''

## import modules
import copy
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


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
                    data_number = transition_value['feature_norm_x'].shape[-1]
                    # Shuffles the indices
                    idx = np.arange(data_number)
                    np.random.shuffle(idx)
                    train_idx = idx[: int(data_number)]
                    # shuffle the data
                    transition_value['feature_x'], transition_value['feature_norm_x'], transition_value['feature_int_y'], transition_value[
                        'feature_onehot_y'] = transition_value['feature_x'][:, :, :, train_idx], transition_value['feature_norm_x'][:, :, :,
                    train_idx], transition_value['feature_int_y'][train_idx], transition_value['feature_onehot_y'][train_idx, :]
    return shuffled_groups
