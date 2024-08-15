##
import numpy as np
from Bipolar_Recognition.Utility_Functions import Emg_Preprocessing, Manipulate_Channels
from Transition_Prediction.Pre_Processing.Utility_Functions import Feature_Calculation


## hdsemg shift
hdsemg_configurations = {
    'hdsemg_original': list(range(1, 12)) + list(range(14, 25)) + list(range(27, 38)) + list(range(40, 51)),
    'hdsemg_h_shift': list(range(14, 25)) + list(range(27, 38)) + list(range(40, 51)) + list(range(53, 64)),  # horizontal shift
    'hdsemg_v_shift': list(range(2, 13)) + list(range(15, 26)) + list(range(28, 39)) + list(range(41, 52))  # vertical shift
}


## save hdsemg features
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']

for subject in subjects:
    hdsemg_features = Emg_Preprocessing.readFeatures(subject, 'hdsemg')

    # select hdsemg features
    for hdsemg_number, channels_to_select in hdsemg_configurations.items():
        selected_features = Manipulate_Channels.extractHdsemgShiftFeatures(hdsemg_features, channels_to_select)

        # save features
        feature_set = hdsemg_number  # there may be multiple sets of features to be calculated for comparison
        Emg_Preprocessing.saveFeatures(subject, selected_features, feature_set)


## bipolar shift
bipolar_configurations = {
    'bipolar_original': [2, 10, 19, 28, 36, 45],
    'bipolar_h_shift_1': [2, 10, 19, 28, 36, 58],
    'bipolar_h_shift_3': [2, 10, 19, 41, 49, 58],
    'bipolar_h_shift_4': [2, 10, 32, 41, 49, 58],
    'bipolar_h_shift_6': [15, 23, 32, 41, 49, 58],
    'bipolar_v_shift_1': [2, 10, 19, 28, 36, 46],
    'bipolar_v_shift_3': [2, 10, 19, 29, 37, 46],
    'bipolar_v_shift_4': [2, 10, 20, 29, 37, 46],
    'bipolar_v_shift_6': [3, 11, 20, 29, 37, 46]
}


## save bipolar emg features
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']
modes = {'standing': 'SS', 'level': 'LW', 'upstairs': 'SA', 'downstairs': 'SD', 'upslope': 'RA', 'downslope': 'RD'}

for subject in subjects:
    emg_preprocessed = Emg_Preprocessing.preprocessEmgData(subject, modes)

    # select bipolar data
    for bipolar_number, channels_to_select in bipolar_configurations.items():
        bipolar_emg = Manipulate_Channels.selectBipolarChannels(emg_preprocessed, channels_to_select)

        # reorganize by sliding windows
        emg_windowed = {}
        for mode, name in modes.items():
            emg_windowed[name] = Emg_Preprocessing.createWindows(bipolar_emg[name], 512, 64)

        # calculate features for each window data
        emg_features = {}
        for mode, name in modes.items():
            emg_feature_list = []
            for emg_window_data in emg_windowed[name]:
                emg_feature = Feature_Calculation.calcuEmgFeatures(emg_window_data)
                emg_feature_list.append(emg_feature)
            emg_features[name] = np.vstack(emg_feature_list).tolist()  # convert numpy to list for dict storage

        # save features
        feature_set = bipolar_number  # there may be multiple sets of features to be calculated for comparison
        Emg_Preprocessing.saveFeatures(subject, emg_features, feature_set)





