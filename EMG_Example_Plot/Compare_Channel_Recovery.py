## import
from Pre_Processing import Preprocessing
from Models.Utility_Functions import Data_Preparation
from Channel_Number.Functions import Channel_Manipulation
import numpy as np
import matplotlib.pyplot as plt


##  read sensor data and filtering
# basic information
subject = 'Number1'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 4, 5, 6, 7, 8, 9, 10]
down_up_session = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
sessions = [up_down_session, down_up_session]


##  define windows
down_sampling = True
start_before_toeoff_ms = 450
endtime_after_toeoff_ms = 400
feature_window_ms = 350
predict_window_ms = start_before_toeoff_ms
sample_rate = 1 if down_sampling is True else 2
predict_window_size = predict_window_ms * sample_rate
feature_window_size = feature_window_ms * sample_rate
predict_window_increment_ms = 20
feature_window_increment_ms = 20
predict_window_shift_unit = int(predict_window_increment_ms / feature_window_increment_ms)
predict_using_window_number = int((predict_window_size - feature_window_size) / (feature_window_increment_ms * sample_rate)) + 1
predict_window_per_repetition = int((endtime_after_toeoff_ms + start_before_toeoff_ms - predict_window_ms) / predict_window_increment_ms) + 1


##  lose certain channels only for testing
channel_random_lost_5 = [3, 10, 32, 46, 50]
channel_random_lost_10 = [2, 12, 19, 22, 26, 43, 46, 49, 54, 64]
channel_random_lost_15 = [3, 7, 8, 16, 19, 23, 27, 32, 38, 39, 41, 46, 56, 59, 63]
channel_random_lost_20 = [1, 4, 6, 12, 13, 14, 15, 18, 22, 28, 31, 35, 39, 43, 46, 48, 51, 55, 60, 62]
# channel_random_lost_25 = [9, 37, 36, 25, 26, 46, 61, 20, 14, 8, 57, 2, 62, 16, 35, 44, 45, 31, 52, 29, 54, 41, 23, 58, 47]

channel_corner_lost_5_upper = [0, 13, 26, 39, 52]
channel_corner_lost_10_upper = [0, 1, 2, 3, 13, 14, 15, 26, 27, 39]
channel_corner_lost_15_upper = [0, 1, 2, 3, 4, 13, 14, 15, 16, 26, 27, 28, 39, 40, 52]
channel_corner_lost_20_upper = [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 26, 27, 28, 29, 39, 40, 41, 52, 53]

channel_corner_lost_5_bottom = [12, 25, 38, 51, 64]
channel_corner_lost_10_bottom = [25, 38, 51, 64, 37, 50, 63, 49, 62, 61]
channel_corner_lost_15_bottom = [12, 24, 25, 36, 37, 38, 48, 49, 50, 51, 60, 61, 62, 63, 64]
channel_corner_lost_20_bottom = [11, 12, 23, 24, 25, 35, 36, 37, 38, 47, 48, 49, 50, 51, 59, 60, 61, 62, 63, 64]


##  read and filter data
split_parameters = Preprocessing.readSplitParameters(subject, version)
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters,
    start_position=-int(start_before_toeoff_ms * (2 / sample_rate)), end_position=int(endtime_after_toeoff_ms * (2 / sample_rate)),
    notchEMG=False, median_filtering=True, reordering=True)
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
del emg_filtered_data
fold = 5  # 5-fold cross validation
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_preprocessed)
del emg_preprocessed


##  recover 20 random lost channels
channel_lost = channel_random_lost_20
random_emg_channel_lost = Channel_Manipulation.losingSomeTestChannels(cross_validation_groups, channel_lost)
random_emg_inpainted = Channel_Manipulation.inpaintImages(random_emg_channel_lost, is_median_filtering=False)  # recover the lost channels (inpaint + median filtering)
random_original = np.transpose(np.reshape(cross_validation_groups['group_0']['test_set']['emg_LWSA'][0], (-1, 13, 5, 2), 'F'), (0, 3, 1, 2))
random_lost_transposed = np.transpose(random_emg_channel_lost['group_0']['test_set']['emg_LWSA'][0], (0, 3, 1, 2))
random_inpaint_transposed = np.transpose(random_emg_inpainted['group_0']['test_set']['emg_LWSA'][0], (0, 3, 1, 2))
random_original_normal = (random_original - np.mean(random_original, axis=0, keepdims=True)) / np.std(random_inpaint_transposed, axis=0, keepdims=True)
random_lost_normal = (random_lost_transposed - np.mean(random_lost_transposed, axis=0, keepdims=True)) / np.std(random_inpaint_transposed, axis=0, keepdims=True)
random_inpaint_normal = (random_inpaint_transposed - np.mean(random_inpaint_transposed, axis=0, keepdims=True)) / np.std(random_inpaint_transposed, axis=0, keepdims=True)


##  recover 20 corner lost channels
channel_lost = channel_corner_lost_20_upper
corner_emg_channel_lost = Channel_Manipulation.losingSomeTestChannels(cross_validation_groups, channel_lost)
corner_emg_inpainted = Channel_Manipulation.inpaintImages(corner_emg_channel_lost, is_median_filtering=False)  # recover the lost channels (inpaint + median filtering)
corner_original = np.transpose(np.reshape(cross_validation_groups['group_0']['test_set']['emg_LWSA'][0], (-1, 13, 5, 2), 'F'), (0, 3, 1, 2))
corner_lost_transposed = np.transpose(corner_emg_channel_lost['group_0']['test_set']['emg_LWSA'][0], (0, 3, 1, 2))
corner_inpaint_transposed = np.transpose(corner_emg_inpainted['group_0']['test_set']['emg_LWSA'][0], (0, 3, 1, 2))
corner_original_normal = (corner_original - np.mean(corner_original, axis=0, keepdims=True)) / np.std(corner_inpaint_transposed, axis=0, keepdims=True)
corner_lost_normal = (corner_lost_transposed - np.mean(corner_lost_transposed, axis=0, keepdims=True)) / np.std(corner_inpaint_transposed, axis=0, keepdims=True)
corner_inpaint_normal = (corner_inpaint_transposed - np.mean(corner_inpaint_transposed, axis=0, keepdims=True)) / np.std(corner_inpaint_transposed, axis=0, keepdims=True)


## create a figure with four subplot of four hdemg heatmaps
timestamp = 600
muscle = 0
random_original_image = random_original_normal[timestamp, muscle, :, :]
random_emg_lost_image = random_lost_normal[timestamp, muscle, :, :]
random_emg_inpainted_image = random_inpaint_normal[timestamp, muscle, :, :]
corner_original_image = corner_original_normal[timestamp, muscle, :, :]
corner_emg_lost_image = corner_lost_normal[timestamp, muscle, :, :]
corner_emg_inpainted_image = corner_inpaint_normal[timestamp, muscle, :, :]

fig, axs = plt.subplots(2, 3, figsize=(10, 10))

# Plot the data in each subplot
for ax, data in zip(axs.flatten(), [random_emg_lost_image, random_original_image, random_emg_inpainted_image, corner_emg_lost_image,
    corner_original_image, corner_emg_inpainted_image]):
    im = ax.imshow(data, cmap='jet')
    fig.colorbar(im, ax=ax)
    # Add text annotations to each grid
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.annotate("{:.2f}".format(data[i, j]), xy=(j, i), ha='center', va='center', color='black', fontsize=6)

# set the title of each subplot
axs[0, 0].set_title("random_lost_emg")
axs[0, 1].set_title("random_inpainted_emg")
axs[0, 2].set_title("random_original_emg")
axs[1, 0].set_title("corner_lost_emg")
axs[1, 1].set_title("corner_inpainted_emg")
axs[1, 2].set_title("corner_original_image")

fig.suptitle("HDEMG images before and after recovery")

