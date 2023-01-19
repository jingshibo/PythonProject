'''
classify using a basic cnn model, get the majority vote results with or without prior information.
'''

## import modules
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Models.ConvRNN.Functions import CnnGru_Model, ConvLstm_Dataset
import datetime

## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 2  # which feature set to use
fold = 5  # 5-fold cross validation

# read feature data
emg_features, emg_feature_2d = Data_Preparation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Preparation.removeSomeSamples(emg_feature_2d)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[-1]  # how many windows there are for each event repetition

# reorganize data
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_feature_data)
normalized_groups = ConvLstm_Dataset.combineNormalizedDataset(cross_validation_groups, window_per_repetition)
shuffled_groups = ConvLstm_Dataset.shuffleTrainingSet(normalized_groups)


## classify using a single CNN-RNN model
now = datetime.datetime.now()
model_results = CnnGru_Model.classifyCnnGruLastOneModel(shuffled_groups)
print(datetime.datetime.now() - now)
# the "many to one" CNN-RNN model does not need majority vote method because each repetition brings only one classification result
accuracy_without_prior = []
for result in model_results:
    accuracy_without_prior.append(result['predict_accuracy'])
# calculate average accuracy without prior information
print('average accuracy without prior:', sum(accuracy_without_prior) / len(accuracy_without_prior))


## results using prior information (no majority vote used, what we need here is the grouped accuracy calculation)
reorganized_results = MV_Results_ByGroup.regroupModelResults(model_results)
accuracy_without_prior, cm = MV_Results_ByGroup.getAccuracyPerGroup(reorganized_results)
average_accuracy, overall_accuracy, sum_cm = MV_Results_ByGroup.averageAccuracy(accuracy_without_prior, cm)
cm_recall = MV_Results_ByGroup.confusionMatrix(sum_cm, recall=True)
print(overall_accuracy)

