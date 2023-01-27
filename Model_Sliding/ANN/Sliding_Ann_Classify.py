## import modules
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Models.ANN.Functions import Ann_Model, Ann_Dataset
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
import datetime


## read emg data
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 0  # which feature set to use
fold = 5  # 5-fold cross validation
feature_window_increment_ms = 16  # the window increment for feature calculation


## read feature data
emg_features, emg_feature_2d = Data_Preparation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Preparation.removeSomeSamples(emg_features)  # only remove some modes here
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_feature_data)

# get shuffled cross validation data set
window_per_repetition = cross_validation_groups['group_0']['train_set']['emg_LWLW_features'][0].shape[0]  # how many windows for each event repetition
normalized_groups = Ann_Dataset.combineNormalizedDataset(cross_validation_groups, window_per_repetition)
shuffled_groups = Ann_Dataset.shuffleTrainingSet(normalized_groups)


## classify using a single ann model
now = datetime.datetime.now()
models, model_results = Ann_Model.classifyUsingAnnModel(shuffled_groups)
print(datetime.datetime.now() - now)
# save model results
##
result_set = 'sliding_ann_0'
Sliding_Ann_Results.saveModelResults(subject, model_results, version, result_set)


## majority vote results using prior information, with a sliding windows to get predict results at different delay points
reorganized_results = MV_Results_ByGroup.regroupModelResults(model_results)
predict_window_shift_unit = 2
sliding_majority_vote_by_group = Sliding_Ann_Results.majorityVoteResultsByGroup(reorganized_results, window_per_repetition, initial_start=0,
    initial_end=16)
accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
# calculate the accuracy and cm. Note: the first dimension refers to each delay
average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
    cm_bygroup)
accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
    predict_window_shift_unit)


## save results




