'''
    The commented area are only used to select an example pair of hdemg and bipolar emg, after which the example data are saved.
    You only need to load them for plotting without the need to run the commented area again.
'''

##
from Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset
from Models.Utility_Functions import Data_Preparation
from EMG_Example_Plot.Utility_Functions import Save_Results
import matplotlib.pyplot as plt
import numpy as np
import copy


##  read sensor data and filtering
# basic information
subject = 'Number1'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 4, 5, 6, 7, 8, 9, 10]
down_up_session = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
sessions = [up_down_session, down_up_session]


# ##  define windows
# down_sampling = True
# start_before_toeoff_ms = 450
# endtime_after_toeoff_ms = 400
# feature_window_ms = 350
# predict_window_ms = start_before_toeoff_ms
# sample_rate = 1 if down_sampling is True else 2
# predict_window_size = predict_window_ms * sample_rate
# feature_window_size = feature_window_ms * sample_rate
# predict_window_increment_ms = 20
# feature_window_increment_ms = 20
# predict_window_shift_unit = int(predict_window_increment_ms / feature_window_increment_ms)
# predict_using_window_number = int((predict_window_size - feature_window_size) / (feature_window_increment_ms * sample_rate)) + 1
# predict_window_per_repetition = int((endtime_after_toeoff_ms + start_before_toeoff_ms - predict_window_ms) / predict_window_increment_ms) + 1
#
# #  read and filter data
# split_parameters = Preprocessing.readSplitParameters(subject, version)
# emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters,
#     start_position=-int(start_before_toeoff_ms * (2 / sample_rate)), end_position=int(endtime_after_toeoff_ms * (2 / sample_rate)),
#     notchEMG=False, reordering=True, median_filtering=True)
# emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
#
# #  build hdemg dataset
# fold = 5  # 5-fold cross validation
# del emg_filtered_data
# cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_preprocessed)
# del emg_preprocessed
# sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(cross_validation_groups, feature_window_size,
#     increment=feature_window_increment_ms * sample_rate)
# del cross_validation_groups
# normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset)
# del sliding_window_dataset
# hdemg_shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
# del normalized_groups
#
#
# ##  build bipolar emg dataset
# channel_selected = [31, 33]
# hdemg_test_data = np.transpose(hdemg_shuffled_groups['group_0']['test_feature_x'], (3, 2, 0, 1))
# bipolar_test_data = copy.deepcopy(hdemg_test_data)
# bipolar_test_data = bipolar_test_data[:, :, :, [31, 31+65]] - bipolar_test_data[:, :, :, [33, 33+65]]
# hdemg_samples = np.reshape(hdemg_test_data[:, 0, :, :], (-1, 350, 13, 10), "F")
# bipolar_samples = np.round(bipolar_test_data[:, 0, :, :], 4)
#
# # extract a frame of hdemg signal
# sample_number = 0
# timestamp = 0
#
# bipolar_time1_device1 = bipolar_samples[sample_number, timestamp, 0]
# bipolar_time1_device2 = bipolar_samples[sample_number, timestamp, 1]
# hdemg_time1_device1 = hdemg_samples[sample_number, timestamp, :, 0:5]
# hdemg_time1_device2 = hdemg_samples[sample_number, timestamp, :, 5:10]
#
# # Find the indices of the elements that have the same bipolar values val1 and val2 in the third dimension
# idx1_3d = np.where(bipolar_samples[:, :, 0] == bipolar_time1_device1)
# idx2_3d = np.where(bipolar_samples[:, :, 1] == bipolar_time1_device2)
# idx1 = np.stack(idx1_3d, axis=1)
# idx2 = np.stack(idx2_3d, axis=1)
# # Find the indices where idx1 and idx2 have the same values
# indices = np.where((idx1[:, None, :] == idx2).all(axis=2))
# sample_number = idx1[indices[0][1]][0]
# timestamp = idx1[indices[0][1]][1]
#
# # extract another frame of hdemg signal with the same bipolar value as the previous hdemg signal
# bipolar_time2_device1 = bipolar_samples[sample_number, timestamp, 0]
# bipolar_time2_device2 = bipolar_samples[sample_number, timestamp, 1]
# hdemg_time2_device1 = hdemg_samples[sample_number, timestamp, :, 0:5]
# hdemg_time2_device2 = hdemg_samples[sample_number, timestamp, :, 5:10]
#
# # save hdemg and bipoalr emg examples
# hdemg_data = [hdemg_time1_device1, hdemg_time2_device1, hdemg_time1_device2, hdemg_time2_device2]
# bipolar_data = [bipolar_time1_device1, bipolar_time2_device1, bipolar_time1_device2, bipolar_time2_device2]
# feature_type = 1
# Save_Results.saveHdemgWithSameBipolar(subject, version, feature_type, hdemg_data, bipolar_data)


## load hdemg and bipoalr emg examples
feature_type = 0
hdemg_data, bipolar_data = Save_Results.loadFeatureExamples(subject, version, feature_type)


## create a figure with four subplot of four hdemg heatmaps
fig, axs = plt.subplots(1, 4, figsize=(10, 10))

# Plot the data in each subplot
font_size = 20
for ax, data, number in zip(axs.flatten(), hdemg_data, bipolar_data):
    im = ax.imshow(data, cmap='jet', interpolation='bicubic')
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    ax.set_xticks(np.arange(0, 5, 2))
    circle1 = plt.Circle((2, 5), 0.3, color='r', fill=False)
    circle2 = plt.Circle((2, 7), 0.3, color='r', fill=False)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.text(2, 6, number, fontsize=font_size, color='red', ha='center', va='center')
    color_bar = fig.colorbar(im, ax=ax, shrink=0.75)
    color_bar.ax.tick_params(labelsize=font_size)

# Adjust the distance between the subplots
fig.subplots_adjust(wspace=0.5)

# set the title of each subplot
axs[0].set_title("(a) RF signals from LW-LW")
axs[1].set_title("(b) RF signals from LW-SA")
axs[2].set_title("(c) TA signals from LW-LW")
axs[3].set_title("(d) TA signals from LW-SA")

# fig.suptitle("HDsEMG Signals With Same Bipolar EMG Values but Different Muscle Activity")





# ## find the bipolar with the closet shape
# bipolar_test_data = np.transpose(bipolar_shuffled_groups['group_0']['test_feature_x'], (3, 2, 1, 0))
# bipolar_sample_1 = bipolar_test_data[:, 0, 0, :]
# bipolar_sample_2 = bipolar_test_data[:, 0, 1, :]


# ##
# from scipy.spatial.distance import cdist
#
# # create a random array with the same shape as yours
# arr = bipolar_sample_1
#
# # calculate the distance between each row
# distances = cdist(arr, arr)
#
# # sort the distances and get the indices of the two closest rows
# sorted_indices = np.argsort(distances, axis=None)
# closest_indices = np.unravel_index(sorted_indices[:2], distances.shape)[0]
#
# print(f"The two closest rows are {closest_indices}")


# ##  load bipolar and hdemg classification results
# bipolar_results = Sliding_Ann_Results.loadModelResults(subject, version, 'channel_area_2', 'Reduced_Cnn')
# result_set = 0
# hdemg_results = Sliding_Ann_Results.loadModelResults(subject, version, result_set, 'Raw_Cnn2d')
