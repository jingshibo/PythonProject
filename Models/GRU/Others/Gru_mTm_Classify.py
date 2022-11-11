'''
classify using a "many to many" GRU model, get the results with or without prior information.
'''

## import modules
from Models.Utility_Functions import Data_Preparation, MV_Results, MV_Results_ByGroup
from Models.GRU.Functions import Gru_Dataset, Gru_Model
import datetime


## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use
fold = 5  # 5-fold cross validation

# read feature data
emg_features, emg_feature_2d = Data_Preparation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Preparation.removeSomeMode(emg_features)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[0]  # how many windows there are for each event repetition
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_feature_data)

# reorganize data
normalized_groups = Gru_Dataset.combineNormalizedDataset(cross_validation_groups, window_per_repetition)
shuffled_groups = Gru_Dataset.shuffleTrainingSet(normalized_groups)

## classify using a "many to many" GRU model
now = datetime.datetime.now()
model_results = Gru_Model.classifyGtuSequenceModel(shuffled_groups)
print(datetime.datetime.now() - now)
# majority vote results
majority_results = MV_Results.majorityVoteResults(model_results, window_per_repetition)
average_accuracy, sum_cm = MV_Results.averageAccuracy(majority_results)
cm_recall = MV_Results.confusionMatrix(sum_cm, recall=True)
print(cm_recall, '\n', average_accuracy)


## majority vote results using prior information
reorganized_results = MV_Results_ByGroup.reorganizeModelResults(model_results)
majority_results = MV_Results_ByGroup.majorityVoteResults(reorganized_results, window_per_repetition)
accuracy_without_prior, cm = MV_Results_ByGroup.getAccuracyPerGroup(reorganized_results)
average_accuracy, overall_accuracy, sum_cm = MV_Results_ByGroup.averageAccuracy(accuracy_without_prior, cm)
cm_recall = MV_Results_ByGroup.confusionMatrix(sum_cm, recall=True)
print(overall_accuracy)


