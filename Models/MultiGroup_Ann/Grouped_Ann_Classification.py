'''
Using prior information to group the data into four categories. For each category using a separate ann model to classify.
'''


## import modules
from Models.Basic_Ann.Functions import Ann_Dataset
from Models.MultiGroup_Ann.Functions import Multiple_Ann_Model, Grouped_Ann_Dataset
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
emg_feature_data = Ann_Dataset.removeSomeMode(emg_features)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[0]  # how many windows there are for each event repetition
cross_validation_groups = Ann_Dataset.crossValidationSet(fold, emg_feature_data)

# reorganize data
transition_grouped = Grouped_Ann_Dataset.separateGroups(cross_validation_groups)
combined_groups = Grouped_Ann_Dataset.combineIntoDataset(transition_grouped, window_per_repetition)
normalized_groups = Grouped_Ann_Dataset.normalizeDataset(combined_groups)
shuffled_groups = Grouped_Ann_Dataset.shuffleTrainingSet(normalized_groups)

## classify using multiple ann models
now = datetime.datetime.now()
model_results = Multiple_Ann_Model.classifyMultipleAnnModel(shuffled_groups)
print(datetime.datetime.now() - now)
majority_results = Multiple_Ann_Model.majorityVoteResults(model_results, window_per_repetition)
accuracy, cm = Multiple_Ann_Model.getAccuracyPerGroup(majority_results)
average_accuracy, sum_cm = Multiple_Ann_Model.averageAccuracy(accuracy, cm)
cm_recall = Multiple_Ann_Model.confusionMatrix(sum_cm, recall=True)
# mean accuracy for all groups
overall_accuracy = (average_accuracy['transition_LW'] * 1.5 + average_accuracy['transition_SA'] + average_accuracy['transition_SD'] +
                    average_accuracy['transition_SS']) / 4.5
print(overall_accuracy)

## save trained models
# type = 1  # define the type of trained model to save
# for number, model in enumerate(model_results):
#     model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models_{type}\cross_validation_set_{number}'
#     if os.path.isfile(model_dir) is False:  # check if the location is a file or directory. Only save when it is a directory
#         model['model'].save(model_dir)  # save the model to the directory

