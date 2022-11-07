'''
classify using a basic cnn model, get the majority vote results with or without prior information.
'''

## import modules
from Models.Utility_Functions import Data_Generation, MV_Results, MV_Results_ByGroup
from Models.CNN.Functions import Cnn_Dataset, Cnn_Model
import datetime

## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use
fold = 5  # 5-fold cross validation

# read feature data
emg_features, emg_feature_2d = Data_Generation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Generation.removeSomeMode(emg_feature_2d)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[-1]  # how many windows there are for each event repetition

# reorganize data
cross_validation_groups = Data_Generation.crossValidationSet(fold, emg_feature_data)
normalized_groups = Cnn_Dataset.combineNormalizedDataset(cross_validation_groups, window_per_repetition)
shuffled_groups = Cnn_Dataset.shuffleTrainingSet(normalized_groups)


## classify using a single cnn model
now = datetime.datetime.now()
model_results = Cnn_Model.classifyUsingCnnModel(shuffled_groups)
print(datetime.datetime.now() - now)

# majority vote results
majority_results = MV_Results.majorityVoteResults(model_results, window_per_repetition)
average_accuracy, sum_cm = MV_Results.averageAccuracy(majority_results)
cm_recall = MV_Results.confusionMatrix(sum_cm, recall=True)
print(cm_recall, '\n', average_accuracy)


## majority vote results using prior information
reorganized_results = MV_Results_ByGroup.reorganizeModelResults(model_results)
majority_results = MV_Results_ByGroup.majorityVoteResults(reorganized_results, window_per_repetition)
accuracy, cm = MV_Results_ByGroup.getAccuracyPerGroup(majority_results)
average_accuracy, sum_cm = MV_Results_ByGroup.averageAccuracy(accuracy, cm)
cm_recall = MV_Results_ByGroup.confusionMatrix(sum_cm, recall=True)

# mean accuracy for all groups
overall_accuracy = (average_accuracy['transition_LW'] * 1.5 + average_accuracy['transition_SA'] + average_accuracy['transition_SD'] +
                    average_accuracy['transition_SS']) / 4.5  # ## save trained models
print(overall_accuracy)

