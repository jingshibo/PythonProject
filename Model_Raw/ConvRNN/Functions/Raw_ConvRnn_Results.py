##
import copy
import numpy as np
from Transition_Prediction.Models.Utility_Functions import MV_Results_ByGroup
from Transition_Prediction.Model_Sliding.ANN.Functions import Sliding_Ann_Results


## get the predict results at each delay time point
def getSlidingResults(reorganized_results, predict_window_per_repetition):
    sliding_results_by_group = copy.deepcopy(reorganized_results)
    for group_value in sliding_results_by_group:
        for transition_type, transition_value in group_value.items():
            for key, value in transition_value.items():
                if key == 'true_value' or key == 'predict_value':
                    transition_value[key] = np.reshape(value, (-1, predict_window_per_repetition), "C")

    return sliding_results_by_group


## majority vote and get the predict results at each delay time point
def SlidingMvResultsByGroup(reorganized_results, predict_using_window_number, predict_window_per_repetition):
    sliding_majority_vote_by_group = copy.deepcopy(reorganized_results)
    for group_value in sliding_majority_vote_by_group:
        for transition_type, transition_value in group_value.items():
            transition_value.pop('predict_softmax', None)  # remove the 'redict_softmax' key
            for key, value in transition_value.items():
                if key == 'true_value' or key == 'predict_value':
                    reshaped_value = np.reshape(value, (-1, predict_using_window_number), "C")
                    majority_vote_results = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=reshaped_value)
                    transition_value[key] = np.reshape(majority_vote_results, (-1, predict_window_per_repetition), "C")

    return sliding_majority_vote_by_group


##  read the model results from disk
def getPredictResults(subject, version, result_set, model_type):
    model_results = Sliding_Ann_Results.loadModelResults(subject, version, result_set, model_type)
    feature_window_increment_ms = model_results[0]['window_parameters']['feature_window_increment_ms']
    predict_window_shift_unit = model_results[0]['window_parameters']['predict_window_shift_unit']
    predict_window_per_repetition = model_results[0]['window_parameters']['predict_window_per_repetition']

    reorganized_results = MV_Results_ByGroup.groupedModelResults(model_results)
    sliding_results_by_group = getSlidingResults(reorganized_results, predict_window_per_repetition)
    accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_results_by_group)
    # calculate the accuracy and cm. Note: the first dimension refers to each delay
    average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(
        accuracy_bygroup, cm_bygroup)
    accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
        predict_window_shift_unit)

    subject_results = {'accuracy': accuracy, 'cm_call': cm_recall}

    return subject_results