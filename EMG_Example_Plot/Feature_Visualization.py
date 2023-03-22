'''
    The commented area are only used to calculate the cnn/manual features for the first time. Then the feature data are saved.
    You only need to load them for plotting without the need to run the commented area again.
'''


## import
from Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset, Raw_Cnn2d_Model
from Models.Utility_Functions import Data_Preparation
from Models.ANN.Functions import Ann_Dataset
from EMG_Example_Plot.Utility_Functions import PCA_Features
import datetime
import numpy as np
import torch
import os


##  read sensor data and filtering
# basic information
subject = 'Number1'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 4, 5, 6, 7, 8, 9, 10]
down_up_session = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
sessions = [up_down_session, down_up_session]


# ## obtain cnn extracted emg feature data (you only need to run this once)
# #  define windows
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
# # read and filter data
# split_parameters = Preprocessing.readSplitParameters(subject, version)
# emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters,
#     start_position=-int(start_before_toeoff_ms * (2 / sample_rate)), end_position=int(endtime_after_toeoff_ms * (2 / sample_rate)),
#     notchEMG=False, reordering=False, median_filtering=True)  # median filtering is necessary to avoid all zero values in a channel
# emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
# del emg_filtered_data
# fold = 5  # 5-fold cross validation
# cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_preprocessed)
# del emg_preprocessed
#
# #  reorganize data
# now = datetime.datetime.now()
# sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(cross_validation_groups, feature_window_size,
#     increment=feature_window_increment_ms * sample_rate)
# del cross_validation_groups
# normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset)
# del sliding_window_dataset
# shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
# del normalized_groups
# print(datetime.datetime.now() - now)
#
# #  classify using a single cnn 2d model
# num_epochs = 50
# batch_size = 1024
# decay_epochs = 20
# now = datetime.datetime.now()
# train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size, report_period=10)
# models, model_results = train_model.trainModel(shuffled_groups, decay_epochs)
# print(datetime.datetime.now() - now)
#
# # get intermediate features from group_0
# trained_parameters = models[0].state_dict()  # get trained model parameters
# input_data = shuffled_groups['group_0']
# # get intermediate CNN features from group_0 only
# cnn_labels, flatten_cnn_features, cnn_features_2d = train_model.extractIntermediateFeatures(input_data, trained_parameters)
# flatten_cnn_features = np.vstack(flatten_cnn_features)
# cnn_labels = np.concatenate(cnn_labels)
# feature_example_set = 1
# feature_type = 'cnn'
# PCA_Feature.saveFeatureExamples(subject, version, feature_example_set, flatten_cnn_features, cnn_labels, feature_type)  # save features


## load the cnn feature examples
feature_example_set = 0
feature_type = 'cnn'
cnn_features, cnn_labels = PCA_Features.loadFeatureExamples(subject, version, feature_example_set, feature_type)
indices = np.where(np.isin(cnn_labels, [4, 5, 6]))[0]  # only keep the indices of three selected modes
cnn_feature_to_pca = cnn_features[indices.tolist(), :]
cnn_labels_to_pca = cnn_labels[indices.tolist()]

##  cnn features pca calculation
cnn_feature_after_pca = PCA_Features.calculatePca(cnn_feature_to_pca, dimension=3)



## obtain manually extracted emg feature data (you only need to run this once)
# # basic information
# subject = "Number1"
# version = 0  # which experiment data to process
# feature_set = 0  # which feature set to use
# fold = 5  # 5-fold cross validation
# # read feature data
# emg_features, emg_feature_2d = Data_Preparation.loadEmgFeature(subject, version, feature_set)
# emg_feature_data = Data_Preparation.removeSomeSamples(emg_features)  # only remove some modes here
# cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_feature_data)
# feature_window_per_repetition = cross_validation_groups['group_0']['train_set']['emg_LWLW_features'][0].shape[0]  # how many windows for each event repetition
# normalized_groups = Ann_Dataset.combineNormalizedDataset(cross_validation_groups, feature_window_per_repetition)
# shuffled_groups = Ann_Dataset.shuffleTrainingSet(normalized_groups)
# # process manually extracted features
# manual_features = shuffled_groups['group_0']['test_feature_x']
# manual_labels = shuffled_groups['group_0']['test_int_y']
# feature_example_set = 1
# feature_type = 'manual'
# PCA_Feature.saveManualFeatures(subject, version, feature_example_set, manual_features, manual_labels, feature_type)  # save features


## load the mannual feature examples
feature_example_set = 0
feature_type = 'manual'
manual_features, manual_labels = PCA_Features.loadFeatureExamples(subject, version, feature_example_set, feature_type)
indices = np.where(np.isin(manual_labels, [4, 5, 6]))[0]  # only keep the indices of three selected modes
manual_features_to_pca = manual_features[indices.tolist(), :]
manual_labels_to_pca = manual_labels[indices.tolist()]

##  manual features pca calculation
manual_feature_after_pca = PCA_Features.calculatePca(manual_features_to_pca, dimension=3)



##  scatter plot reduced-dimensional data
PCA_Features.plotPcaFeatures(cnn_feature_after_pca, cnn_labels_to_pca, manual_feature_after_pca, manual_labels_to_pca)


