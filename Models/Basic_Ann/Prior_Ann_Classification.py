'''
classify using a basic 4 layer ann model. get the majority vote results (within groups) with prior information
'''

## import modules
from Models.Basic_Ann.Functions import Prior_Ann_Model, CV_Dataset
from Models.MultiGroup_Ann.Functions import Multiple_Ann_Model
import datetime
import os


## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use

# read feature data
emg_features, emg_feature_reshaped = CV_Dataset.loadEmgFeature(subject, version, feature_set)
emg_feature_data = CV_Dataset.removeSomeMode(emg_features)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[0]  # how many windows there are for each event repetition

# get shuffled cross validation data set
fold = 5  # 5-fold cross validation
cross_validation_groups = CV_Dataset.crossValidationSet(fold, emg_feature_data)
normalized_groups = CV_Dataset.combineIntoDataset(cross_validation_groups, window_per_repetition)
shuffled_groups = CV_Dataset.shuffleTrainingSet(normalized_groups)


## classify using an ann model
now = datetime.datetime.now()
model_results = Prior_Ann_Model.classifyPriorAnnModel(shuffled_groups)
print(datetime.datetime.now() - now)

##
reorganized_results = Prior_Ann_Model.reorganizeModelResults(model_results)
majority_results = Multiple_Ann_Model.majorityVoteResults(reorganized_results, window_per_repetition)
accuracy, cm = Multiple_Ann_Model.getAccuracyPerGroup(majority_results)
average_accuracy, sum_cm = Multiple_Ann_Model.averageAccuracy(accuracy, cm)
cm_recall = Multiple_Ann_Model.confusionMatrix(sum_cm, recall=True)


## mean accuracy for all groups
overall_accuracy = (average_accuracy['transition_LW'] * 1.5 + average_accuracy['transition_SA'] + average_accuracy['transition_SD'] +
                    average_accuracy['transition_SS']) / 4.5  # ## save trained models
print(overall_accuracy)

# ## save trained models
# type = 1  # define the type of trained model to save
# for number, model in enumerate(model_results):
#     model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models_{type}\cross_validation_set_{number}'
#     if os.path.isfile(model_dir) is False:  # check if the location is a file or directory. Only save when it is a directory
#         model['model'].save(model_dir)  # save the model to the directory

