'''
Using prior information to group the data into four categories. For each category using a separate ann model to classify.
'''


## import modules
from Models.Basic_Ann.Functions import CV_Dataset
from Models.MultiGroup_Ann.Functions import Multiple_Ann_Model, Grouped_CV_Dataset
import datetime
import os
import tensorflow as tf
import numpy as np


## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use

# read feature data
emg_features, emg_feature_reshaped = CV_Dataset.loadEmgFeature(subject, version, feature_set)
emg_feature_data = CV_Dataset.removeSomeMode(emg_features)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[0]  # how many windows there are for each event repetition
fold = 5  # 5-fold cross validation
cross_validation_groups = CV_Dataset.crossValidationSet(fold, emg_feature_data)

# reorganize data
transition_grouped = Grouped_CV_Dataset.separateGroups(cross_validation_groups)
combined_groups = Grouped_CV_Dataset.combineIntoDataset(transition_grouped, window_per_repetition)
normalized_groups = Grouped_CV_Dataset.normalizeDataset(combined_groups)
shuffled_groups = Grouped_CV_Dataset.shuffleTrainingSet(normalized_groups)

## classify using multiple ann models
now = datetime.datetime.now()
model_results = Multiple_Ann_Model.classifyMultipleAnnModel(shuffled_groups)
print(datetime.datetime.now() - now)
majority_results = Multiple_Ann_Model.majorityVoteResults(model_results, window_per_repetition)
accuracy, cm = Multiple_Ann_Model.getAccuracyPerGroup(majority_results)
average_accuracy, sum_cm = Multiple_Ann_Model.averageAccuracy(accuracy, cm)
cm_recall = Multiple_Ann_Model.confusionMatrix(sum_cm, recall=True)
print(average_accuracy, cm_recall)

# ## save trained models
# type = 1  # define the type of trained model to save
# for number, model in enumerate(model_results):
#     model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models_{type}\cross_validation_set_{number}'
#     if os.path.isfile(model_dir) is False:  # check if the location is a file or directory. Only save when it is a directory
#         model['model'].save(model_dir)  # save the model to the directory

