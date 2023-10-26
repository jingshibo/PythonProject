#
# ## recover emg signal (both deleting duplicated data and add missing data)
# # how to deal with duplicated data: delete all these data and use the first data and the last data as reference to count the number of
# # missing data, then insert Nan of this number
# now = datetime.datetime.now()
# # delete duplicated data
# start_index = 588863  # the index before the first duplicated data
# end_index = 590748  # the index after the last duplicated data.
# # Note: it only drop data between start_index+1 ~ end_index-1. the last index will not be dropped.
# dropped_emg_data = raw_emg_data.drop(raw_emg_data.index[range(start_index+1, end_index)])
# inserted_emg_data = Insole_Emg_Recovery.insertEmgNanRow(dropped_emg_data, start_index, end_index)
#
# # how to deal with lost data: calculate the number of missing data and then insert Nan of the same number
# start_index = 588775  # the last index before the missing data
# end_index = 588776  # the first index after the missing data
# inserted_emg_data = Insole_Emg_Recovery.insertEmgNanRow(inserted_emg_data, start_index, end_index)
#
# # # interpolate emg data
# reindex_emg_data = inserted_emg_data.sort_index().reset_index(drop=True)  # Reorder DataFrame
# recovered_emg_data = reindex_emg_data.interpolate(method='cubic', limit_direction='forward', axis=0)
# print(datetime.datetime.now() - now)
#
#
#
#
# ##
# # Final Refined Logic:
# # If the absolute value of the difference is in the hundreds or thousands:
# # Positive Difference: Classify as missing data.
# # Negative Difference: Classify as duplicated data.
# # If the absolute value of the difference is close to 60000:
# # This likely indicates an anomaly occurring around the time of a wraparound.
# # Negative Difference: Classify as missing data (since it occurred at the wraparound, the missing data makes it negative).
# # Positive Difference: Classify as duplicated data (since it occurred at the wraparound, the duplicated data makes it positive).
#
# # Function to insert NaN rows at specific indices
# def insert_nan_rows(df, indices):
#     nan_row = pd.Series([float('nan')]*df.shape[1], index=df.columns)
#     for index in indices:
#         df.loc[index] = nan_row
#     return df.sort_index().reset_index(drop=True)
#
# # Step 1: Remove Duplicated Data
# import copy
# emg = copy.deepcopy(raw_emg_data)
#
# # Identify start points of duplicated segments using wrong_timestamp
# duplicate_start_points = wrong_timestamp_emg.index[
#     (wrong_timestamp_emg['count_difference'].abs().between(100, 9999) & (wrong_timestamp_emg['count_difference'] < 0)) | (
#                 wrong_timestamp_emg['count_difference'].abs() >= 59000) & (wrong_timestamp_emg['count_difference'] > 0)].tolist()
# # Identify duplicated data based on the refined logic
#
# # Drop the rows corresponding to duplicated data
# df_cleaned = emg.loc[~duplicated_data_mask].reset_index(drop=True)
#
# # Step 2: Identify Missing Data
#
# # Re-calculate differences between adjacent timestamps in the cleaned DataFrame
# df_cleaned['diff'] = df_cleaned['timestamp'].diff()
#
# # Identify missing data based on the refined logic
# missing_data_mask = ((df_cleaned['diff'].abs() <= 9999 & (df_cleaned['diff'] > 0)) |
#                       (df_cleaned['diff'].abs() >= 59000) & (df_cleaned['diff'] < 0))
#
# # Get indices where missing data is identified
# missing_data_indices = df_cleaned.index[missing_data_mask].tolist()
#
# # Insert NaN rows at the missing data indices
# df_with_nans = insert_nan_rows(df_cleaned, missing_data_indices)
#
# # Step 3: Interpolate Missing Data
#
# # Interpolate missing data using cubic method
# df_interpolated = df_with_nans.interpolate(method='cubic')
#
#
#
## preprocess raw emg data for recovery
# ## Function to identify and remove all duplicated data based on each starting point of duplication
# def identify_and_remove_all_duplicates(raw_emg_data, duplicate_start_points):
#     raw_emg = copy.deepcopy(raw_emg_data)
#     # List to store the indices of rows to remove
#     rows_duplicated = []
#
#     # Name of the last column, which is assumed to contain the timestamps
#     timestamp_col = raw_emg.columns[-1]
#
#     # Loop through the list of starting points
#     for start_row_index in duplicate_start_points:
#         start_timestamp = raw_emg.iloc[start_row_index, -1]
#
#         # Find the closest point where duplication begins by tracing back to previous timestamps
#         closest_duplication_start = raw_emg[raw_emg.index < start_row_index].loc[raw_emg[timestamp_col] == start_timestamp].index.max()
#
#         # Initialize variables to track the expected timestamp and the end point of duplication
#         expected_timestamp = start_timestamp
#         end_row_index = None
#
#         # Identify all duplicated data from the starting point until the duplication ends
#         for row_index in range(start_row_index, len(raw_emg)):
#             if raw_emg.iloc[row_index, -1] == expected_timestamp:
#                 end_row_index = row_index  # Update the end point of the duplicated segment
#                 expected_timestamp += 1  # Update the expected timestamp for the next iteration
#             else:
#                 break  # Exit the loop if the data no longer match
#
#         # Add the indices of the duplicated rows to the list
#         rows_duplicated.extend(range(closest_duplication_start, end_row_index + 1))
#
#     # Remove the duplicated rows
#     df_cleaned = raw_emg.drop(rows_duplicated).reset_index(drop=True)
#
#     return df_cleaned, rows_duplicated
#
# df_cleaned, rows_duplicated = identify_and_remove_all_duplicates(raw_emg_data, duplicate_start_points)
#



##
from Transition_Prediction.Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset, Raw_Cnn2d_Model, Model_Storage
from Transition_Prediction.Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Transition_Prediction.Model_Sliding.ANN.Functions import Sliding_Ann_Results
import datetime


##  read sensor data and filtering
# basic information
subject = 'Number1'
version = 0  # the data from which experiment version to process
modes = ['up_down_t0', 'down_up_t0']
up_down_session = [0, 1, 2, 3, 4]
down_up_session = [1, 2, 3, 4, 5]
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


## read and filter data
split_parameters = Preprocessing.readSplitParameters(subject, version, project='cGAN_Model')
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters,
    start_position=-int(start_before_toeoff_ms * (2 / sample_rate)), end_position=int(endtime_after_toeoff_ms * (2 / sample_rate)),
    notchEMG=False, reordering=True, median_filtering=True, project='cGAN_Model')  # median filtering is necessary to avoid all zero values in a channel
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
# del emg_filtered_data
fold = 5  # 5-fold cross validation
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_preprocessed)
# del emg_preprocessed


##  reorganize data
now = datetime.datetime.now()
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(cross_validation_groups, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
# del cross_validation_groups
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize='z-score', limit=1500)
# del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
# del normalized_groups
print(datetime.datetime.now() - now)


##  classify using a single cnn 2d model
num_epochs = 50
batch_size = 1024
decay_epochs = 20
now = datetime.datetime.now()
# train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size)
train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size, report_period=10)
models, model_results = train_model.trainModel(shuffled_groups, decay_epochs)
print(datetime.datetime.now() - now)


## save model results
model_type = 'Raw_Cnn2d'
result_set = 2
window_parameters = {'predict_window_ms': predict_window_ms, 'feature_window_ms': feature_window_ms, 'sample_rate': sample_rate,
    'predict_window_increment_ms': predict_window_increment_ms, 'feature_window_increment_ms': feature_window_increment_ms,
    'predict_window_shift_unit': predict_window_shift_unit, 'predict_using_window_number': predict_using_window_number,
    'start_before_toeoff_ms': start_before_toeoff_ms, 'endtime_after_toeoff_ms': endtime_after_toeoff_ms,
    'predict_window_per_repetition': predict_window_per_repetition, 'feature_window_per_repetition': feature_window_per_repetition}
Sliding_Ann_Results.saveModelResults(subject, model_results, version, result_set, window_parameters, model_type)


## save models
model_name = list(range(fold))
Model_Storage.saveModels(models, subject, version, model_type, model_name, project='Insole_Emg')


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

