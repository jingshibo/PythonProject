##
from Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Channel_Number.Functions import Channel_Manipulation, Reduced_Cnn1d_Model, Reduced_Cnn2d_Model
import datetime
import numpy as np

##  read sensor data and filtering
# basic information
subject = 'Number5'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
down_up_session = [0, 1, 2, 3, 4, 5, 6, 8, 9]
# up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# down_up_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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


## manipulate channels
# select certain channels for model training
channel_density_33 = list(range(0, 65, 2))
channel_density_21 = list(range(0, 13, 2)) + list(range(26, 39, 2)) + list(range(52, 65, 2))
channel_density_11 = list(range(0, 13, 4)) + list(range(28, 39, 4)) + list(range(52, 65, 4))
channel_density_8 = list(range(0, 13, 4)) + list(range(52, 65, 4))

channel_area_35 = list(range(3, 10)) + list(range(16, 23)) + list(range(29, 36)) + list(range(42, 49)) + list(range(55, 62))
channel_area_25 = list(range(4, 9)) + list(range(17, 22)) + list(range(30, 35)) + list(range(43, 48)) + list(range(56, 61))
channel_area_15 = list(range(17, 22)) + list(range(30, 35)) + list(range(43, 48))
channel_area_6 = [31, 32, 33, 44, 45, 46]

channel_muscle_hdemg1 = list(range(0, 65))
channel_muscle_hdemg2 = list(range(65, 130))
channel_muscle_bipolar1 = [31]
channel_muscle_bipolar2 = [96]
channel_muscle_bipolar = [31, 33]


## read and filter data
split_parameters = Preprocessing.readSplitParameters(subject, version)
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters,
    start_position=-int(start_before_toeoff_ms * (2 / sample_rate)), end_position=int(endtime_after_toeoff_ms * (2 / sample_rate)),
    notchEMG=False, median_filtering=True, reordering=False)
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)


## select part of the channels
channel_selected = channel_area_6
result_set = 'channel_area_6'
emg_channel_selected = Channel_Manipulation.selectSomeChannels(emg_preprocessed, channel_selected)
# del emg_filtered_data
fold = 5  # 5-fold cross validation
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_channel_selected)
# del emg_preprocessed


##  only for bipolar emg on one muscle
if len(channel_selected) == 1:
    for group_number, group_value in cross_validation_groups.items():
        for set_type, set_value in group_value.items():
            for transition_label, transition_data in set_value.items():
                for repetition_number, repetition_data in enumerate(transition_data):
                    transition_data[repetition_number] = repetition_data[:, np.newaxis]


##  reorganize data
now = datetime.datetime.now()
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(cross_validation_groups, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
# del cross_validation_groups
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
shuffled_data = shuffled_groups
# shuffled_data = {'group_0': shuffled_groups['group_0'], 'group_3': shuffled_groups['group_3']}
if len(channel_selected) > 2:  # for hdemg of different channel numbers
    train_model = Reduced_Cnn2d_Model.ModelTraining(num_epochs, batch_size, report_period=10)
    models, model_results = train_model.trainModel(shuffled_data, decay_epochs)
else:  # for bipolar emg only
    train_model = Reduced_Cnn1d_Model.ModelTraining(num_epochs, batch_size, report_period=10)
    models, model_results = train_model.trainModel(shuffled_data, decay_epochs)

print(datetime.datetime.now() - now)


## save model results
model_type = 'Reduced_Cnn'
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

