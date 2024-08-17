##
import numpy as np
import pandas as pd
from scipy.ndimage import shift
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Bipolar_Recognition.Utility_Functions import Emg_Preprocessing
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


## extract features from selected hdsemg channels
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


## shift a single image by a random amount
def shiftEmgImage(image, max_shift, direction):
    # Shifts an image by a random amount along either axes.
    if direction == 'up':
        shift_x = -np.random.randint(0, max_shift + 1)
        shift_y = 0
    elif direction == 'down':
        shift_x = np.random.randint(0, max_shift + 1)
        shift_y = 0
    elif direction == 'left':
        shift_y = -np.random.randint(0, max_shift + 1)
        shift_x = 0
    elif direction == 'right':
        shift_y = np.random.randint(0, max_shift + 1)
        shift_x = 0
    else:
        raise Exception('wrong direction!')

    # Shift the image and fill the empty areas with 0
    shifted_image = shift(image, shift=[shift_x, shift_y, 0], mode='constant', cval=0)

    return shifted_image


## shift a bunch of images for multiple times
def multipleShifts(interp_emg_features, direction, max_shift, num_shifts, vertical_channels, horizontal_channels):
    shift_clip_emg = {f'shift_{i + 1}': {} for i in range(num_shifts)}  # Dynamically create keys for each shift

    for i in range(num_shifts):  # Loop over the number of shifts
        for locomotion_mode, locomotion_values in interp_emg_features.items():
            # Horizontal shift to the left by 8 pixels
            shift_emg = np.array([shiftEmgImage(image, max_shift, direction) for image in locomotion_values])

            # Clip only the central part of each shifted image
            clip_images = shift_emg[:, vertical_channels, horizontal_channels, :]

            # Save the results under the corresponding shift key for the current locomotion mode
            repetition = f'shift_{i + 1}'  # 'shift_1', 'shift_2', ..., 'shift_n'
            shift_clip_emg[repetition][locomotion_mode] = [clip_images[i] for i in range(clip_images.shape[0])]  # change to a list

    return shift_clip_emg


## clip and shift original emg image dataset
def clipShiftimages(interp_emg_features, shift_direction, max_shift=8, num_shifts=3, vertical_channels=slice(8, 89),
        horizontal_channels=slice(0, 25)):
    # clip original images, keep only the central parts
    clip_original_emg = {}  # no shift, only truncate
    for locomotion_mode, locomotion_values in interp_emg_features.items():
        feature_values = locomotion_values[:, vertical_channels, horizontal_channels, :]
        clip_original_emg[locomotion_mode] = [feature_values[i] for i in range(feature_values.shape[0])]  # convert to a list for later process

    # shift and clip emg feature images for multiple times
    clip_shift_emg = multipleShifts(interp_emg_features, direction=shift_direction, max_shift=max_shift, num_shifts=num_shifts,
        vertical_channels=vertical_channels, horizontal_channels=horizontal_channels)

    return clip_original_emg, clip_shift_emg


## construct cross validation set with augmented shift data
def constructAugmentDataset(clip_original_emg, clip_shift_emg):
    emg_cross_validation_original = Data_Preparation.crossValidationSet(5, clip_original_emg, shuffle=False)
    emg_cross_validation_shift = {}
    for shift_repetition, shift_values in clip_shift_emg.items():
        emg_cross_validation_shift[shift_repetition] = Data_Preparation.crossValidationSet(5, shift_values, shuffle=False)

    # add the train_set data in the emg_cross_validation_shift dict into the train_set in the emg_cross_validation_original dict
    for shift_key, shift_data in emg_cross_validation_shift.items():
        for group_key, group_data in shift_data.items():
            # Access the corresponding 'train_set' in both the original and shift data
            train_set_shift = group_data['train_set']
            train_set_original = emg_cross_validation_original[group_key]['train_set']

            # Move the data for each key in the 'train_set'
            for key in train_set_shift:
                # Extend the original list with the data from the shift set
                train_set_original[key].extend(train_set_shift[key])

            # Clear the moved train_set data to free memory
            group_data['train_set'] = None
        # Clear the moved shift data to free memory
        emg_cross_validation_shift[shift_key] = None
    # Finally, delete emg_cross_validation_shift to free up the memory
    del emg_cross_validation_shift

    return emg_cross_validation_original
