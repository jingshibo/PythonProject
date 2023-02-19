##
from Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset, Raw_Cnn2d_Model
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
import datetime


##  read sensor data and filtering
# basic information
subject = 'Number5'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
down_up_session = [0, 1, 2, 3, 4, 5, 6, 8, 9]
# up_down_session = [2, 3, 4, 5, 6, 7, 12, 13, 14]
# down_up_session = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sessions = [up_down_session, down_up_session]


## read and filter data
split_parameters = Preprocessing.readSplitParameters(subject, version)
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-900,
    end_position=800, notchEMG=False, reordering=True)
down_sampling = True
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
del emg_filtered_data
fold = 5  # 5-fold cross validation
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_preprocessed)
del emg_preprocessed


##  define windows
predict_window_ms = 450
feature_window_ms = 350
sample_rate = 1 if down_sampling is True else 2
predict_window_size = predict_window_ms * sample_rate
feature_window_size = feature_window_ms * sample_rate
predict_window_increment_ms = 20
feature_window_increment_ms = 20
predict_window_shift_unit = int(predict_window_increment_ms / feature_window_increment_ms)
predict_of_window_number = int((predict_window_size - feature_window_size) / (feature_window_increment_ms * sample_rate))
endtime_after_toeoff_ms = 400
predict_window_per_repetition = int(endtime_after_toeoff_ms / predict_window_increment_ms) + 1
window_parameters = {'predict_window_ms': predict_window_ms, 'feature_window_ms': feature_window_ms, 'sample_rate': sample_rate,
    'predict_window_increment_ms': predict_window_increment_ms, 'feature_window_increment_ms': feature_window_increment_ms,
    'predict_window_shift_unit': predict_window_shift_unit, 'predict_of_window_number': predict_of_window_number,
    'endtime_after_toeoff_ms': endtime_after_toeoff_ms, 'predict_window_per_repetition': predict_window_per_repetition}



##  reorganize data
now = datetime.datetime.now()
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(cross_validation_groups, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
del cross_validation_groups
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups
print(datetime.datetime.now() - now)


##  classify using a single cnn 2d model
num_epochs = 55
batch_size = 1024
decay_epochs = 20
now = datetime.datetime.now()
# train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size)
train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size, report_period=10)
models, model_results = train_model.trainModel(shuffled_groups, decay_epochs)
print(datetime.datetime.now() - now)


## save model results
result_set = 0
model_type = 'Raw_Cnn2d'
Sliding_Ann_Results.saveModelResults(subject, model_results, version, result_set, window_parameters, model_type)


## majority vote results using prior information, with a sliding windows to get predict results at different delay points
reorganized_results = MV_Results_ByGroup.regroupModelResults(model_results)
sliding_majority_vote_by_group = Sliding_Ann_Results.majorityVoteResultsByGroup(reorganized_results, feature_window_per_repetition,
    predict_window_shift_unit, predict_of_window_number, initial_start=0)
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

