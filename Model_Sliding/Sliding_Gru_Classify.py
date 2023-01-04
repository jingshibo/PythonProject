## import modules
from Models.Utility_Functions import Data_Preparation
from Model_Sliding.Functions import Sliding_Gru_Dataset, Sliding_Gru_Model, Sliding_Results_ByGroup, Sliding_Evaluation_ByGroup
import datetime

## read emg data
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 0  # which feature set to use
fold = 5  # 5-fold cross validation

# read feature data
emg_features, emg_feature_2d = Data_Preparation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Preparation.removeSomeSamples(emg_features)  # only remove some modes here
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_feature_data)

# create dataset
now = datetime.datetime.now()
emg_sliding_features = Sliding_Gru_Dataset.createSlidingDataset(cross_validation_groups, initial_start=0, initial_end=16, shift=4)
window_per_repetition = emg_sliding_features['group_0']['train_set']['emg_LWLW_features'][0].shape[0]  # how many windows for each event repetition
normalized_groups = Sliding_Gru_Dataset.combineNormalizedDataset(emg_sliding_features, window_per_repetition)
shuffled_groups = Sliding_Gru_Dataset.shuffleTrainingSet(normalized_groups)
print(datetime.datetime.now() - now)

## classify using a "many to one" GRU model
now = datetime.datetime.now()
model_results = Sliding_Gru_Model.classifySlidingGtuLastOneModel(shuffled_groups)
print(datetime.datetime.now() - now)

## predict results using sliding windows (category + delay)
# regroup the model results using prior information (no majority vote used, what we need here is the grouped accuracy calculation)
regrouped_results = Sliding_Results_ByGroup.regroupModelResults(model_results)
# reorganize the regrouped model results based on the timestamps
reorganized_softmax, reorganized_prediction, reorganized_truevalues = Sliding_Results_ByGroup.reorganizePredictValues(regrouped_results)
#  find the first timestamps at which the softmax value is larger than the threshold
first_timestamps = Sliding_Results_ByGroup.findFirstTimestamp(reorganized_softmax, threshold=0.995)
# query the prediction based on timestamps from the reorganized_prediction table
sliding_prediction = Sliding_Results_ByGroup.getSlidingPredictResults(reorganized_prediction, first_timestamps, increment=16, shift=4)

## evaluate the prediction accuracy
accuracy_bygroup, cm_bygroup = Sliding_Evaluation_ByGroup.getAccuracyPerGroup(sliding_prediction, reorganized_truevalues)
average_accuracy, overall_accuracy, sum_cm = Sliding_Evaluation_ByGroup.averageAccuracy(accuracy_bygroup, cm_bygroup)
cm_recall = Sliding_Evaluation_ByGroup.confusionMatrix(sum_cm, recall=True)
print(overall_accuracy)

## evaluate the prediction delay
import numpy as np
import copy

integrated_information = copy.deepcopy(reorganized_truevalues)
for group_number, group_value in enumerate(sliding_prediction):
    for transition_type, transition_value in group_value.items():
        # put true and predict information into a single array (order: predict_category, true_category, category_difference, delay_value)
        combined_array = np.array([transition_value[0, :], reorganized_truevalues[group_number][transition_type],
            reorganized_truevalues[group_number][transition_type] - transition_value[0, :], transition_value[1, :]])
        unequal_columns = np.where(combined_array[2, :] != 0)[0]  # the column index where the predict values differ from the true values

        # remove the columns with the true values and predict values that are unequal
        if unequal_columns.size > 0:
            integrated_information[group_number][transition_type] = np.delete(combined_array, unequal_columns, 1)
        else:
            integrated_information[group_number][transition_type] = combined_array
        # only keep the true_value row and delay_value row
        integrated_information[group_number][transition_type] = np.delete(integrated_information[group_number][transition_type], [0, 2], 0)

## calculate the mean and std values of delay for each category
a = np.array([])
for group_value in integrated_information:
    for transition_type, transition_value in group_value.items():
        a = np.concatenate((a, transition_value), axis=1)



## only calculate the delay for those with correct prediction results
import numpy as np
x = np.array([1,0,2,2,3,3,4,5,6,7,8])
a = np.where(x != 0)[0]
# a = combined_array[2, :]


## the problem is that it takes up a lot memories
# import numpy as np
# import tensorflow as tf
# length3 = np.random.uniform(0, 1, size=(3, 2))
# length4 = np.random.uniform(0, 1, size=(4, 2))
# tf.keras.preprocessing.sequence.pad_sequences([length3, length4], dtype='float32', padding='post')
