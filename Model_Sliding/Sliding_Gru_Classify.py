## import modules
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.Functions import Sliding_Gru_Dataset, Sliding_Gru_Model, Sliding_Results_ByGroup
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


##
import matplotlib.pyplot as plt
accuracy = []
for result in model_results:
    for shift_number, shift_value in result.items():
        accuracy.append(shift_value['predict_accuracy'])
plt.plot(accuracy)


## results using sliding windows
# regroup the model results using prior information (no majority vote used, what we need here is the grouped accuracy calculation)
regrouped_results = Sliding_Results_ByGroup.regroupModelResults(model_results)
# reorganize the regrouped model results based on the timestamps
reorganized_softmax, reorganized_prediction, reorganized_truevalues = Sliding_Results_ByGroup.reorganizePredictValues(regrouped_results)


##
import numpy as np
import copy
threshold = 0.995
first_timestamps_above_threshold = copy.deepcopy(reorganized_softmax)
for group_value in first_timestamps_above_threshold:
    for transition_type, transition_value in group_value.items():
        #  for each repetition, find the first timestamp at which the softmax value is larger than the threshold
        largest_softmax = np.transpose(np.amax(transition_value, axis=2))  # for each timestamp, return the largest softmax value among all categories
        first_timestamp = np.argmax(largest_softmax > threshold, axis=0)  # for each repetition, return the first timestamp at which the softmax value is above the threshold

        # dedicated addressing the special case when a repetition has all the softmax values (from all timestamps) below the threshold
        first_softmax = largest_softmax[first_timestamp.tolist(), list(range(largest_softmax.shape[1]))]  # find the first softmax value above the threshold for each repetition
        threshold_test = first_softmax / threshold  # calculate the ratio to show if there are any repetitions with all the softmax values below the threshold
        repetition_low_softmax = np.transpose(np.squeeze(np.argwhere(threshold_test < 1)))  # return the index of the repetition with all softmax values below the threshold
        first_timestamp[repetition_low_softmax.tolist()] = -1  # change the timestamp index  to -1, for the repetition with all softmax values below the threshold

        group_value[transition_type] = first_timestamp


## query the prediction based on timestamps from the reorganized_prediction table
delay_unit = 16 * 4  # 16 is the window increment value(ms), 4 is the shift number
sliding_prediction = copy.deepcopy(reorganized_prediction)
for group_number, group_value in enumerate(first_timestamps_above_threshold):
    for transition_type, transition_value in group_value.items():
        # get the prediction results based on the timestamp information
        predict_result = sliding_prediction[group_number][transition_type][transition_value.tolist(), list(range(transition_value.shape[0]))]
        # predict result AND delay value (the first row is prediction results, the second row is the delay value)
        sliding_prediction[group_number][transition_type] = np.array([predict_result, transition_value * delay_unit])

        # dedicated addressing the special case when the delay value is negative (due to the -1 timestamp value assigned)
        negative_indices = np.argwhere(sliding_prediction[group_number][transition_type][1, :] < 0)
        # negative delay value means there is no results with probabilities higher than the threshold in this repetition
        if negative_indices.size > 0:
            for column in np.nditer(negative_indices):
                prediction = np.bincount(reorganized_prediction[group_number][transition_type][:, column]).argmax()  # use the most common prediction as the result
                delay_value = delay_unit * (reorganized_prediction[group_number][transition_type].shape[0] - 1)  # use the maximum delay value instead
                sliding_prediction[group_number][transition_type][0, column] = prediction
                sliding_prediction[group_number][transition_type][1, column] = delay_value


##
a = np.array([[11, 12, 13, 14, 15, 16, 17, 15],
                [13, 19, 14, 15, 16, 17, 18, 19]])
for x in np.nditer(negative_indices):
    print(x, end=' ')

##
accuracy_with_prior, cm_with_prior = Sliding_Results_ByGroup.getAccuracyPerGroup(regrouped_results)
average_accuracy, overall_accuracy, sum_cm = Sliding_Results_ByGroup.averageAccuracy(accuracy_with_prior, cm_with_prior)
cm_recall = MV_Results_ByGroup.confusionMatrix(sum_cm, recall=True)
print(overall_accuracy)




## the problem is that it takes up a lot memories
# import numpy as np
# import tensorflow as tf
# length3 = np.random.uniform(0, 1, size=(3, 2))
# length4 = np.random.uniform(0, 1, size=(4, 2))
# tf.keras.preprocessing.sequence.pad_sequences([length3, length4], dtype='float32', padding='post')
