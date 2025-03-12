##
import numpy as np
from HDsEMG_Recognition.Utility_Functions import Emg_Preprocessing, Manipulate_Channels
from Transition_Prediction.Pre_Processing.Utility_Functions import Feature_Calculation


## select bipolar channels
bipolar_selection = {
    'bipolar_1': [32],
    'bipolar_2': [29, 35],
    'bipolar_3': [28, 32, 36],
    'bipolar_4': [16, 22, 42, 48],
    'bipolar_5': [15, 23, 32, 41, 49],
    'bipolar_6': [3, 9, 29, 35, 55, 61],
    'bipolar_7': [6, 15, 23, 32, 41, 49, 58],
    'bipolar_8': [2, 6, 10, 29, 35, 54, 58, 62],
    'bipolar_9': [2, 6, 10, 28, 32, 36, 54, 58, 62]}


## load emg data
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']
modes = {'standing': 'SS', 'level': 'LW', 'upstairs': 'SA', 'downstairs': 'SD', 'upslope': 'RA', 'downslope': 'RD'}

for subject in subjects:
    emg_preprocessed = Emg_Preprocessing.preprocessEmgData(subject, modes)

    # select bipolar data
    for bipolar_number, channels_to_select in bipolar_selection.items():
        bipolar_emg = Manipulate_Channels.selectBipolarChannels(emg_preprocessed, channels_to_select)

        ## reorganize by sliding windows
        emg_windowed = {}
        for mode, name in modes.items():
            emg_windowed[name] = Emg_Preprocessing.createWindows(bipolar_emg[name], 512, 64)

        ## calculate features for each window data
        emg_features = {}
        for mode, name in modes.items():
            emg_feature_list = []
            for emg_window_data in emg_windowed[name]:
                emg_feature = Feature_Calculation.calcuEmgFeatures(emg_window_data)
                emg_feature_list.append(emg_feature)
            emg_features[name] = np.vstack(emg_feature_list).tolist()  # convert numpy to list for dict storage

        ## save features
        feature_set = bipolar_number  # there may be multiple sets of features to be calculated for comparison
        Emg_Preprocessing.saveFeatures(subject, emg_features, feature_set)
