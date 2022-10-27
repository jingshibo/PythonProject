## import modules
from Models.Utility_Functions import Cross_Validation, Ann_Model


## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use

# read feature data
emg_features, emg_feature_reshaped = Cross_Validation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Cross_Validation.removeSomeMode(emg_features)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[0]  # how many windows there are for each event repetition

# get shuffled cross validation data set
fold = 5  # 5-fold cross validation
cross_validation_groups = Cross_Validation.crossValidationSet(fold, emg_feature_data)
normalized_groups = Cross_Validation.combineIntoDataset(cross_validation_groups, window_per_repetition)
shuffled_groups = Cross_Validation.shuffleTrainingSet(normalized_groups)

## classify using an ann model
classification_results = Ann_Model.classifyUsingAnnModel(shuffled_groups)
majority_results = Ann_Model.majorityVoteResults(classification_results, window_per_repetition)
mean_accuracy, sum_cm = Ann_Model.averageAccuracy(majority_results)
cm_recall = Ann_Model.confusionMatrix(sum_cm, recall=True)
print(cm_recall, '\n', mean_accuracy)
