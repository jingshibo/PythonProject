import pandas as pd
import numpy as np

## insert a electrode at 0 position
def insertElectrode(emg_data):
    # insert 0 channel
    emg_data.insert(0, "blank_column", "")
    emg_data.columns = list(range(0, 65))  # column name
    # emg_data.loc[:, 0] = emg_data.loc[:, [1, 24, 25]].mean(axis=1)
    emg_data[0] = emg_data[[1, 24, 25]].mean(axis=1)
    return emg_data


## adjust electrode order to match the physical EMG grid
def reorderElectrodes(emg_data):
    table1 = emg_data.loc[:, 0:12]
    table2 = emg_data.loc[:, 13:25].iloc[:, ::-1]  # reverse the electrodes between 13 and 25
    table3 = emg_data.loc[:, 26:38]
    table4 = emg_data.loc[:, 39:51].iloc[:, ::-1]  # reverse the electrodes between 39 and 51
    table5 = emg_data.loc[:, 52:64]
    emg_reordered = pd.concat([table1, table2, table3, table4, table5], axis=1)  # reorder the emg data
    emg_reordered.columns = list(range(0, 65))  # column name
    return emg_reordered


## rearrange emg data to map 4d matrix
def reshapeEmgFeatures(emg_feature_data):
    emg_feature_reshaped = {}
    emg_repetitions = []
    rows = 13
    columns = 5
    features = -1
    for gait_event_label, gait_event_emg in emg_feature_data.items():
        for repetition_emg_features in gait_event_emg:
            samples = len(repetition_emg_features)
            emg_repetitions.append(np.reshape(np.transpose(repetition_emg_features), (rows, columns, features, samples), order='F'))
        emg_feature_reshaped[gait_event_label] = emg_repetitions
        emg_repetitions = []
    return emg_feature_reshaped