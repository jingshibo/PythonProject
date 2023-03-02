## import
from Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset, Raw_Cnn2d_Model
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Channel_Number.Functions import Channel_Manipulation
import datetime


##  read sensor data and filtering
# basic information
subject = 'Number5'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
# up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# down_up_session = [0, 1, 2, 3, 4, 5, 6, 8, 9]
up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
down_up_session = [0, 1, 2, 3, 4, 5, 6, 8, 9]
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


## lose certain channels only for testing
channel_random_lost_5 = [3, 10, 32, 46, 50]
channel_random_lost_6 = [2, 12, 22, 26, 46, 54]
channel_random_lost_10 = [2, 12, 19, 22, 26, 43, 46, 49, 54, 64]
channel_random_lost_13 = [2, 12, 19, 22, 26, 43, 46, 49, 54, 64, 6, 30, 31]
channel_random_lost_15 = [2, 12, 19, 22, 26, 43, 46, 49, 54, 64, 6, 30, 55, 36]

channel_corner_lost_5_upper = [0, 13, 26, 39, 52]
channel_corner_lost_6_upper = [0, 1, 2, 13, 14, 26]
channel_corner_lost_10_upper = [0, 1, 2, 3, 13, 14, 15, 26, 27, 39]
channel_corner_lost_13_upper = [0, 1, 2, 3, 4, 13, 14, 15, 26, 27, 28, 39, 52]
channel_corner_lost_15_upper = [0, 1, 2, 3, 4, 13, 14, 15, 16, 26, 27, 28, 39, 40, 52]

channel_corner_lost_5_bottom = [12, 25, 38, 51, 64]
channel_corner_lost_6_bottom = [38, 51, 64, 50, 63, 62]
channel_corner_lost_10_bottom = [25, 38, 51, 64, 37, 50, 63, 49, 62, 61]
channel_corner_lost_13_bottom = [12, 25, 38, 51, 64, 37, 50, 63, 36, 49, 62, 61, 60]
channel_corner_lost_15_bottom = [12, 25, 38, 51, 64, 24, 37, 50, 63, 36, 49, 62, 48, 61, 60]


## read and filter data
split_parameters = Preprocessing.readSplitParameters(subject, version)
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters,
    start_position=-int(start_before_toeoff_ms * (2 / sample_rate)), end_position=int(endtime_after_toeoff_ms * (2 / sample_rate)),
    notchEMG=False, median_filtering=False, reordering=True)
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
del emg_filtered_data
fold = 5  # 5-fold cross validation
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_preprocessed)
del emg_preprocessed


##  select and recover losing channels
now = datetime.datetime.now()
channel_lost = channel_random_lost_13
emg_channel_lost = Channel_Manipulation.losingSomeTestChannels(cross_validation_groups, channel_lost)
# del cross_validation_groups
emg_inpainted = Channel_Manipulation.inpaintImages(emg_channel_lost, is_median_filtering=False)  # recover the lost channels (inpaint + median filtering)
emg_recovered, emg_unrecovered = Channel_Manipulation.restoreEmgshape(emg_inpainted, emg_channel_lost)  # restore the shape of an image
del emg_inpainted
print(datetime.datetime.now() - now)


##  reorganize data
data_to_process = emg_recovered
result_set = 'channel_random_lost_13_recovered'
now = datetime.datetime.now()
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(data_to_process, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
# del data_to_process
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups
print(datetime.datetime.now() - now)


##  classify using a single cnn 2d model
num_epochs = 50
batch_size = 1024
decay_epochs = 20
now = datetime.datetime.now()
shuffled_data = {'group_0': shuffled_groups['group_0'], 'group_3': shuffled_groups['group_3']}
# shuffled_data = shuffled_groups
train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size, report_period=10)
models, model_results = train_model.trainModel(shuffled_data, decay_epochs)
print(datetime.datetime.now() - now)


## save model results
model_type = 'Losing_Cnn'
window_parameters = {'predict_window_ms': predict_window_ms, 'feature_window_ms': feature_window_ms, 'sample_rate': sample_rate,
    'predict_window_increment_ms': predict_window_increment_ms, 'feature_window_increment_ms': feature_window_increment_ms,
    'predict_window_shift_unit': predict_window_shift_unit, 'predict_using_window_number': predict_using_window_number,
    'endtime_after_toeoff_ms': endtime_after_toeoff_ms, 'predict_window_per_repetition': predict_window_per_repetition,
    'feature_window_per_repetition': feature_window_per_repetition}
Sliding_Ann_Results.saveModelResults(subject, model_results, version, result_set, window_parameters, model_type)


## majority vote results using prior information, with a sliding windows to get predict results at different delay points
reorganized_results = MV_Results_ByGroup.groupedModelResults(model_results)
sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results, feature_window_per_repetition,
    predict_window_shift_unit, predict_using_window_number, initial_start=0)
accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
# calculate the accuracy and cm. Note: the first dimension refers to each delay
average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
    cm_bygroup)
accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
    predict_window_shift_unit)

