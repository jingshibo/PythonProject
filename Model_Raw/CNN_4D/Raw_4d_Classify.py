##
from Transition_Prediction.Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset
from Model_Raw.CNN_4D.Functions import Raw_Cnn4d_Model
from Transition_Prediction.Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Transition_Prediction.Model_Sliding.ANN.Functions import Sliding_Ann_Results
import datetime
import numpy as np


##  read sensor data and filtering
# basic information
subject = 'Shibo'
version = 1  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
down_up_session = [10, 11, 12, 13, 19, 24, 25, 26, 27, 28, 20]
# up_down_session = [10, 11]
# down_up_session = [10, 11]
sessions = [up_down_session, down_up_session]


## read and filter data
split_parameters = Preprocessing.readSplitParameters(subject, version)
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-1024,
    end_position=1024, notchEMG=False, reordering=True)
down_sampling = True
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
del emg_filtered_data
fold = 5  # 5-fold cross validation
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_preprocessed)
del emg_preprocessed


##  define windows
predict_window_ms = 512
feature_window_ms = 256
sample_rate = 2 if down_sampling is False else 1
predict_window_size = predict_window_ms * sample_rate
feature_window_size = feature_window_ms * sample_rate
predict_window_increment_ms = 32
feature_window_increment_ms = 32
predict_window_shift_unit = int(predict_window_increment_ms / feature_window_increment_ms)
predict_using_window_number = int((predict_window_size - feature_window_size) / (feature_window_increment_ms * sample_rate)) + 1


##  reorganize data
now = datetime.datetime.now()
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(cross_validation_groups, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
# del cross_validation_groups
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups
# reshape the emg data to match the physical grid
for group_number, group_value in shuffled_groups.items():
    group_value['train_feature_x'] = np.reshape(group_value['train_feature_x'], (feature_window_size, 13, 10, 1, -1), "F")
    group_value['test_feature_x'] = np.reshape(group_value['test_feature_x'], (feature_window_size, 13, 10, 1, -1), "F")
print(datetime.datetime.now() - now)


##  classify using a single cnn 2d model
num_epochs = 55
batch_size = 1024
decay_epochs = 20
now = datetime.datetime.now()
# train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size)
train_model = Raw_Cnn4d_Model.ModelTraining(num_epochs, batch_size, report_period=10)
models, model_results = train_model.trainModel(shuffled_groups, decay_epochs)
print(datetime.datetime.now() - now)


## save model results
result_set = 0
Sliding_Ann_Results.saveModelResults(subject, model_results, version, result_set, feature_window_per_repetition,
    feature_window_increment_ms, predict_window_shift_unit, model_type='Raw_CNN4D')

## majority vote results using prior information, with a sliding windows to get predict results at different delay points
reorganized_results = MV_Results_ByGroup.groupedModelResults(model_results)
sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results, feature_window_per_repetition,
    predict_window_shift_unit, initial_start=0, predict_using_window_number=predict_using_window_number)
accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
# calculate the accuracy and cm. Note: the first dimension refers to each delay
average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
    cm_bygroup)
accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
    predict_window_shift_unit)



# import os
# import json
#
# model_type = 'Raw_Cnn2d'
# result_set = 1
#
# for group_number, group_value in shuffled_groups.items():
#     for key, value in group_value.items():
#         group_value[key] = value.tolist()
#
# data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\extracted_features'
# result_file = f'subject_{subject}_Experiment_{version}_emg_raw_data_{result_set}.json'
# result_path = os.path.join(data_dir, result_file)
#
# with open(result_path, 'w') as json_file:
#     json.dump(shuffled_groups, json_file, indent=8)



