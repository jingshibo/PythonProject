'''
Using prior information to group the data into four categories. For each category using a separate ann model to classify.
'''


## import modules
from Models.Basic_Ann.Functions import Ann_Dataset
from Models.MultiGroup_Ann.Functions import Multiple_Ann_Model, Grouped_Ann_Dataset
from Models.CNN.Functions import Grouped_Cnn_Dataset
import datetime
import tensorflow as tf
import os


## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use
fold = 5  # 5-fold cross validation

# read feature data
emg_features, emg_feature_reshaped = Ann_Dataset.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Ann_Dataset.removeSomeMode(emg_feature_reshaped)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[-1]  # how many windows there are for each event repetition
cross_validation_groups = Ann_Dataset.crossValidationSet(fold, emg_feature_data)

# reorganize data
transition_grouped = Grouped_Ann_Dataset.separateGroups(cross_validation_groups)
combined_groups = Grouped_Cnn_Dataset.combineIntoDataset(transition_grouped, window_per_repetition)
normalized_groups = Grouped_Cnn_Dataset.normalizeDataset(combined_groups)
shuffled_groups = Grouped_Cnn_Dataset.shuffleTrainingSet(normalized_groups)


## classify using multiple cnn models

