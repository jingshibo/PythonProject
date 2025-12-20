##
import matplotlib.pyplot as plt
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


## plot the pulses of imu and emg for alignment
def plotImuData(imu_data, column, start_index, end_index):
    # if you mean the third column by position, use iloc
    imu_sync = imu_data.iloc[:, column]

    # extract the portion you want to plot
    segment = imu_sync.iloc[start_index:end_index]

    # create figure + axis
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(range(len(segment)), segment, label="imu_value")
    ax.set(title="imu", ylabel="value")
    ax.tick_params(labelbottom=True)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def print_c_array(name, arr, float_precision=6):
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[0]
    elems = ", ".join(f"{v:.{float_precision}f}f" for v in arr)
    print(f"const float {name}[{n}] = {{ {elems} }};")


def combineNormalizedDataset(cross_validation_groups):
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
                    train_feature_x.extend(np.array(gait_event_features))
                    train_feature_y.extend([gait_event_label] * len(gait_event_features))

            elif set_type == 'test_set':  # keep the structure unchanged
                for gait_event_label, gait_event_features in set_value.items():
                    test_feature_x.extend(np.array(gait_event_features))
                    test_feature_y.extend([gait_event_label] * len(gait_event_features))

        # convert to numpy arrays
        train_feature_x = np.asarray(train_feature_x, dtype=np.float32)
        test_feature_x = np.asarray(test_feature_x, dtype=np.float32)

        # ----- normalization: compute per-feature mean/std on TRAIN only -----
        train_mean = np.mean(train_feature_x, axis=0)
        train_std = np.std(train_feature_x, axis=0)

        # avoid division by zero
        train_std[train_std == 0] = 1.0

        train_norm_x = (train_feature_x - train_mean) / train_std
        test_norm_x = (test_feature_x - train_mean) / train_std

        # ====== NEW: print C arrays for this group's normalization ======
        print(f"\n// ---- Normalization parameters for group {group_number} ----")
        print_c_array(f"feature_mean", train_mean)
        print_c_array(f"feature_std", train_std)
        # ===============================================================

        # Free up space by deleting original unnormalized arrays
        del train_feature_x, test_feature_x
        gc.collect()

        # one-hot encode categories (according to the alphabetical order)
        train_int_y = LabelEncoder().fit_transform(train_feature_y)
        train_onehot_y = tf.keras.utils.to_categorical(train_int_y)

        test_int_y = LabelEncoder().fit_transform(test_feature_y)
        test_onehot_y = tf.keras.utils.to_categorical(test_int_y)

        # put training data and test data into one group
        normalized_groups[group_number] = {"train_feature_x": train_norm_x, "train_int_y": train_int_y, "train_onehot_y": train_onehot_y,
            "test_feature_x": test_norm_x, "test_int_y": test_int_y, "test_onehot_y": test_onehot_y}

    return normalized_groups
