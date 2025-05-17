

##
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import numpy as np


## build classification datasets and extract the relevent modes for data generation
def extractGanTrainingData(modes_generation, old_emg_normalized, new_emg_normalized, time_length):
    # build classification dataset
    all_labels = sorted(set(old_emg_normalized.keys()) | set(new_emg_normalized.keys()))
    label_map = {label: idx for idx, label in enumerate(all_labels)}

    # select only the central part data for training
    def slice_center_time(emg_dict, desired_length):
        new_emg_dict = {}
        for label, samples in emg_dict.items():
            new_emg_dict[label] = []
            for sample in samples:
                total_length = sample.shape[0]
                start = (total_length - desired_length) // 2
                end = start + desired_length
                new_emg_dict[label].append(sample[start:end, :])  # Slice time axis
        return new_emg_dict
    old_emg_central = slice_center_time(old_emg_normalized, time_length)
    new_emg_central = slice_center_time(new_emg_normalized, time_length)

    # build the paired x and y dataset
    def classifier_dataset(emg_dict, label_map, all_labels):
        data = []
        labels_categorical = []

        for label in all_labels:
            if label in emg_dict:
                for sample in emg_dict[label]:
                    data.append(sample[np.newaxis, :, :])  # (1, 2000, 65)
                    labels_categorical.append(label_map[label])

        data = np.array(data)
        labels_categorical = np.array(labels_categorical).reshape(-1, 1)

        encoder = OneHotEncoder(sparse_output=False)
        labels_onehot = encoder.fit_transform(labels_categorical)

        return {'data_x': data, 'int_y': labels_categorical.flatten(), 'onehot_y': labels_onehot, 'label_map': label_map}

    classify_old_emg = classifier_dataset(old_emg_central, label_map, all_labels)
    classify_new_emg = classifier_dataset(new_emg_central, label_map, all_labels)

    # build gan generation dataset
    train_gan_data = {}
    data_keys = ['gen_data_1', 'gen_data_2', 'disc_data']  # The order in the list is critical, corresponding to the locomotion modes

    for transition_type, modes in modes_generation.items():
        # Initialize transition_type key in real_emg and train_gan_data dictionaries
        train_gan_data[transition_type] = {'gen_data_1': None, 'gen_data_2': None, 'disc_data': None}
        for idx, mode in enumerate(modes):
            # Assign values using the new structure
            train_gan_data[transition_type][data_keys[idx]] = old_emg_central[mode]

    return classify_old_emg, classify_new_emg, train_gan_data


## build cross validation datset for classification
def crossValidationSet(fold_number, classify_emg_data):
    # StratifiedKFold ensures that each fold has the same class distribution as the full dataset.
    skf = StratifiedKFold(n_splits=fold_number, shuffle=True, random_state=42)
    X = classify_emg_data['data_x']
    y = classify_emg_data['int_y']

    # only index
    cross_validation_indices = []
    for train_idx, val_idx in skf.split(X, y):
        cross_validation_indices.append({'train': train_idx, 'val': val_idx})

    # fold data
    cross_validation_dataset = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_data = {'train_x': X[train_idx], 'train_y': y[train_idx], 'val_x': X[val_idx], 'val_y': y[val_idx]}
        cross_validation_dataset.append(fold_data)

        print(f"Fold {fold_idx + 1}:")
        unique_train, counts_train = np.unique(fold_data['train_y'], return_counts=True)
        unique_val, counts_val = np.unique(fold_data['val_y'], return_counts=True)
        print(f"  Train class counts: {dict(zip(unique_train, counts_train))}")
        print(f"  Val class counts:   {dict(zip(unique_val, counts_val))}")

    return cross_validation_indices, cross_validation_dataset

