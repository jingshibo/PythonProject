##
import pandas as pd
from Bipolar_Recognition.Utility_Functions import Emg_Preprocessing
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
import copy


## select derived bipolar channels
def selectBipolarChannels(emg_data, selected_channels):
    # Initialize a dictionary to store the bipolar emg results
    bipolar_selection = {}

    for locomotion_mode, locomotion_values in emg_data.items():
        # Initialize a DataFrame to store differences for this key
        bipolar_values = pd.DataFrame()

        # Compute differences for specified channels in the current feature
        for channel in selected_channels:
            if channel < 63:  # Ensure the channel + 2 does not exceed bounds
                # Compute differences between the specified channel and channel + 2
                bipolar_values[f'channel_{channel}_diff'] = locomotion_values.iloc[:, channel-1] - locomotion_values.iloc[:, channel + 1]
            else:
                raise Exception('wrong channel number!')
            # Store the resulting DataFrame of differences in the results dictionary
            bipolar_selection[locomotion_mode] = bipolar_values

    return bipolar_selection


## extract features from certain hdsemg channels
def extractHdsemgShiftFeatures(hdsemg_features, selected_channels):
    # Initialize a dictionary to store the extracted values
    channel_results = {}

    for locomotion_mode, locomotion_values in hdsemg_features.items():
        channel_results[locomotion_mode] = []

        for features in locomotion_values:
            channel_values = []

            # Extract values for each feature
            for i in range(len(features) // 65):
                values = []
                base_index = i * 65  # Start index for each feature's channel data
                # Extract values for specified channels in the current feature type
                for channel in selected_channels:
                    # Ensure the channel is within the bounds of 0 to 64
                    if channel < 65:
                        value = features[base_index + channel]
                        values.append(value)
                channel_values.extend(values)
            channel_results[locomotion_mode].append(channel_values)

    return channel_results


## construct the dataset for shift evaluation with the training set from original features and test set from shift features
def constructShiftDataset(subject, original_features, shift_features):
    # extract original features
    emg_feature_original = Emg_Preprocessing.readFeatures(subject, original_features)
    emg_cross_validation_original = Data_Preparation.crossValidationSet(5, emg_feature_original, shuffle=False)

    # extract shift features
    emg_feature_shift = Emg_Preprocessing.readFeatures(subject, shift_features)
    emg_cross_validation_shift = Data_Preparation.crossValidationSet(5, emg_feature_shift, shuffle=False)

    # use the test set in the shift feature dict to substitute the test set in the original feature dict
    emg_cross_validation = copy.deepcopy(emg_cross_validation_original)
    for group_key in emg_cross_validation_original.keys():
        # Replace 'test_set' in 'c' with 'test_set' from 'b'
        emg_cross_validation[group_key]['test_set'] = copy.deepcopy(emg_cross_validation_shift[group_key]['test_set'])

    return emg_cross_validation