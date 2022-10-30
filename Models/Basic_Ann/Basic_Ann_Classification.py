'''
classify using a basic 4 layer ann model, get the majority vote results (accuracy and confusion matrix), and save the trained models
'''

## import modules
from Models.Basic_Ann.Functions import Ann_Model, CV_Dataset
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
model_results = Ann_Model.classifyUsingAnnModel(shuffled_groups)
print(datetime.datetime.now() - now)
majority_results = Ann_Model.majorityVoteResults(model_results, window_per_repetition)
mean_accuracy, sum_cm = Ann_Model.averageAccuracy(majority_results)
cm_recall = Ann_Model.confusionMatrix(sum_cm, recall=True)
print(cm_recall, '\n', mean_accuracy)


## save trained models
type = 1  # define the type of trained model to save
for number, model in enumerate(model_results):
    model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models_{type}\cross_validation_set_{number}'
    if os.path.isfile(model_dir) is False:  # check if the location is a file or directory. Only save when it is a directory
        model['model'].save(model_dir)  # save the model to the directory
