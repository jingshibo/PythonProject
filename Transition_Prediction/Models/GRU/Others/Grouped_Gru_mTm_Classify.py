'''
Using prior information to group the data into four categories. For each category using a separate "many to many" GRU model to classify.
'''

## import modules
from Transition_Prediction.Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Transition_Prediction.Models.GRU.Functions import Gru_Model, Grouped_Gru_Dataset
import datetime


## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use
fold = 5  # 5-fold cross validation

# read feature data
emg_features, emg_feature_reshaped = Data_Preparation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Preparation.removeSomeSamples(emg_features)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[0]  # how many windows there are for each event repetition
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_feature_data)

# reorganize data
transition_grouped = Data_Preparation.separateGroups(cross_validation_groups)
combined_groups = Grouped_Gru_Dataset.combineIntoDataset(transition_grouped, window_per_repetition)
normalized_groups = Grouped_Gru_Dataset.normalizeDataset(combined_groups, window_per_repetition)
shuffled_groups = Grouped_Gru_Dataset.shuffleTrainingSet(normalized_groups)

## classify using multiple "many to many" GRU model models
now = datetime.datetime.now()
model_results = Gru_Model.classifyMultipleGruSequenceModel(shuffled_groups)
print(datetime.datetime.now() - now)

## majority vote results using prior information
majority_results = MV_Results_ByGroup.majorityVoteResultsByGroup(model_results, window_per_repetition)
accuracy_without_prior, cm = MV_Results_ByGroup.getAccuracyPerGroup(majority_results)
average_accuracy, overall_accuracy, sum_cm = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_without_prior, cm)
cm_recall = MV_Results_ByGroup.confusionMatrix(sum_cm, is_recall=True)
print(overall_accuracy)