## import modules
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.Functions import Sliding_Gru_Dataset, Sliding_Gru_Model, Sliding_Results_ByGroup, Sliding_Evaluation_ByGroup
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## read emg data
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 0  # which feature set to use
fold = 5  # 5-fold cross validation
increment_ms = 16  # the window increment for feature calculation


## read feature data
emg_features, emg_feature_2d = Data_Preparation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Preparation.removeSomeSamples(emg_features)  # only remove some modes here
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_feature_data)
del emg_features, emg_feature_2d, emg_feature_data  # to release memory

# create dataset
now = datetime.datetime.now()
shift_unit = 3   # shift_unit defines the number of window shifts for each classification sliding
emg_sliding_features = Sliding_Gru_Dataset.createSlidingDataset(cross_validation_groups, shift_unit, initial_start=0, initial_end=16)
window_per_repetition = emg_sliding_features['group_0']['train_set']['emg_LWLW_features'][0].shape[0]  # how many windows for each event repetition
normalized_groups = Sliding_Gru_Dataset.combineNormalizedDataset(emg_sliding_features, window_per_repetition)
shuffled_groups = Sliding_Gru_Dataset.shuffleTrainingSet(normalized_groups)
del cross_validation_groups, emg_sliding_features, normalized_groups  # to release memory
print(datetime.datetime.now() - now)


## classify using a "many to one" GRU model
now = datetime.datetime.now()
model_results = Sliding_Gru_Model.classifySlidingGtuLastOneModel(shuffled_groups)
print(datetime.datetime.now() - now)
## in case you want to release more memory
del shuffled_groups  # remove the variable


##  group the classification results together from diffferent initial_predict_time settings
group_sliding_results = Sliding_Evaluation_ByGroup.groupSlidingResults(model_results, shift_unit, increment_ms, threshold=0.99)


## (separately) predict results at a certain initial_predict_time (category + delay)
# regroup the model results using prior information (no majority vote used, what we need here is the grouped accuracy calculation)
regrouped_results = Sliding_Results_ByGroup.regroupModelResults(model_results)
# reorganize the regrouped model results based on the timestamps
reorganized_softmax, reorganized_prediction, reorganized_truevalues = Sliding_Results_ByGroup.reorganizePredictValues(regrouped_results)
# only keep the results after the initial_predict_time
initial_predict_time = 4  # define the initial prediction timestamp from which the classification starts
reduced_softmax, reduced_prediction = Sliding_Results_ByGroup.reducePredictResults(reorganized_softmax, reorganized_prediction, initial_predict_time)
#  find the first timestamps at which the softmax value is larger than the threshold
first_timestamps = Sliding_Results_ByGroup.findFirstTimestamp(reduced_softmax, threshold=0.99)
# get the predict results based on timestamps from the reorganized_prediction table and convert the timestamp to delay(ms)
sliding_prediction = Sliding_Results_ByGroup.getSlidingPredictResults(reduced_prediction, first_timestamps, initial_predict_time, shift_unit, increment_ms)


## evaluate the prediction results at a certain initial_predict_time
# calculate the prediction accuracy
accuracy_bygroup, cm_bygroup = Sliding_Evaluation_ByGroup.getAccuracyPerGroup(sliding_prediction, reorganized_truevalues)
average_accuracy, overall_accuracy, sum_cm = MV_Results_ByGroup.averageAccuracy(accuracy_bygroup, cm_bygroup)
cm_recall = MV_Results_ByGroup.confusionMatrix(sum_cm, recall=True)
print(overall_accuracy)
# calculate the prediction delay (separately for those with correct or false prediction results)
correct_results, false_results = Sliding_Evaluation_ByGroup.integrateResults(sliding_prediction, reorganized_truevalues)
predict_delay_category, predict_delay_overall, mean_delay, std_delay = Sliding_Evaluation_ByGroup.countDelay(correct_results)
false_delay_category, false_delay_overall, false_delay_mean, false_delay_std = Sliding_Evaluation_ByGroup.countDelay(false_results)
predict_delay_overall, false_delay_overall = Sliding_Evaluation_ByGroup.delayAccuracy(predict_delay_overall, false_delay_overall)






# ## save and read shuffled_groups
# import os
# data_set = 0
# data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\model_dataset'
# feature_file = f'subject_{subject}_Experiment_{version}_model_dataset_{data_set}.npy'
# feature_path = os.path.join(data_dir, feature_file)
# ##
# np.save(feature_path, shuffled_groups)
# ##
# shift_unit = 2
# shuffled_groups = np.load(feature_path, allow_pickle=True).item()


## the problem is that it takes up a lot memories
# import numpy as np
# import tensorflow as tf
# length3 = np.random.uniform(0, 1, size=(3, 2))
# length4 = np.random.uniform(0, 1, size=(4, 2))
# tf.keras.preprocessing.sequence.pad_sequences([length3, length4], dtype='float32', padding='post')
