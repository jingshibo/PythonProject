'''
classify using a "many to one" GRU model, get the results with or without prior information.
'''

## import modules
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
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

## classify using a "many to one" GRU model
now = datetime.datetime.now()
model_results = Gru_Model.classifyGtuLastOneModel(shuffled_groups, window_per_repetition)
print(datetime.datetime.now() - now)
# the "many to one" RNN model does not need majority vote method because each repetition brings only one classification result
accuracy_without_prior = []
for result in model_results:
    accuracy_without_prior.append(result['predict_accuracy'])
# calculate average accuracy without prior information
print('average accuracy without prior:', sum(accuracy_without_prior) / len(accuracy_without_prior))

## results using prior information (no majority vote used, what we need here is the grouped accuracy calculation)
reorganized_results = MV_Results_ByGroup.reorganizeModelResults(model_results)
accuracy_without_prior, cm = MV_Results_ByGroup.getAccuracyPerGroup(reorganized_results)
average_accuracy, overall_accuracy, sum_cm = MV_Results_ByGroup.averageAccuracy(accuracy_without_prior, cm)
cm_recall = MV_Results_ByGroup.confusionMatrix(sum_cm, recall=True)
print(overall_accuracy)

