##
import os
import copy
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.linalg import block_diag

##  majority vote results for all transitions without grouping
def majorityVoteResults(model_results, window_per_repetition, initial_start=0, initial_end=16, predict_window_shift_unit=2):
    # reorganize the results
    bin_results = []
    for result in model_results:  # reunite the samples from the same transition
        true_y = []
        predict_y = []
        for key, value in result.items():
            if key == 'true_value':
                for i in range(0, len(value), window_per_repetition):
                    true_y.append(value[i: i+window_per_repetition])
            elif key == 'predict_value':
                for i in range(0, len(value), window_per_repetition):
                    predict_y.append(value[i: i+window_per_repetition])
        bin_results.append({"true_value": true_y, "predict_value": predict_y})

    shift_range = window_per_repetition - initial_start - initial_end  # decide how many shifts to do in the for loop below
    sliding_majority_vote = copy.deepcopy(bin_results)
    for group, result in enumerate(bin_results):
        for key, value in result.items():
            for number, each_repetition in enumerate(value):
                sliding_result_each_repetition = []
                for shift in range(0, shift_range, predict_window_shift_unit):  # reorganize the predict results at each delay timepoint
                    slicing_value = each_repetition[initial_start + shift: initial_end + shift + 1]
                    sliding_result_each_repetition.append(np.bincount(slicing_value).argmax())  # get majority vote results at the delay time
                sliding_majority_vote[group][key][number] = sliding_result_each_repetition
                # convert nested list to numpy: row is the repetition, column is the predict results at each delay time in this repetition
                sliding_majority_vote[group][key] = np.array(sliding_majority_vote[group][key])

    return sliding_majority_vote


##  majority vote results based on transition groups
def majorityVoteResultsByGroup(reorganized_results, window_per_repetition, initial_start=0, initial_end=16, predict_window_shift_unit=2):
    # reunite the samples belonging to the same transition
    bin_results = []
    for each_group in reorganized_results:  # reunite the samples from the same transition
        bin_transitions = {}
        for transition_type, transition_results in each_group.items():
            true_y = []
            predict_y = []
            for key, value in transition_results.items():
                if key == 'true_value':
                    for i in range(0, len(value), window_per_repetition):
                        true_y.append(value[i: i + window_per_repetition])
                elif key == 'predict_value':
                    for i in range(0, len(value), window_per_repetition):
                        predict_y.append(value[i: i + window_per_repetition])
            bin_transitions[transition_type] = {"true_value": true_y, "predict_value": predict_y}
        bin_results.append(bin_transitions)

    # use majority vote to get a consensus result for each repetition
    shift_range = window_per_repetition - initial_start - initial_end  # decide how many shifts to do in the for loop below
    sliding_majority_vote_by_group = copy.deepcopy(bin_results)
    for group, result in enumerate(bin_results):
        for transition_type, transition_results in result.items():
            for key, value in transition_results.items():
                for number, each_repetition in enumerate(value):
                    sliding_result_each_repetition = []
                    for shift in range(0, shift_range, predict_window_shift_unit):  # reorganize the predict results at each delay timepoint
                        slicing_value = each_repetition[initial_start + shift: initial_end + shift + 1]
                        sliding_result_each_repetition.append(np.bincount(slicing_value).argmax())  # get majority vote results at the delay time
                    sliding_majority_vote_by_group[group][transition_type][key][number] = sliding_result_each_repetition
                # convert nested list to numpy: row is the repetition, column is the predict results at each delay time in this repetition
                sliding_majority_vote_by_group[group][transition_type][key] = np.array(sliding_majority_vote_by_group[group][transition_type][key])

    return sliding_majority_vote_by_group


##  calculate accuracy and cm values for each group
def getAccuracyPerGroup(sliding_majority_vote_by_group):
    accuracy_bygroup = []
    cm_bygroup = []
    for each_group in sliding_majority_vote_by_group:
        transition_accuracy = {}
        transition_cm = {}
        for transition_type, transition_result in each_group.items():
            true_y = transition_result['true_value']
            predict_y = transition_result['predict_value']
            numCorrect = np.count_nonzero(true_y == predict_y, axis=0)
            transition_accuracy[transition_type] = numCorrect / true_y.shape[0] * 100
            # loop over the results at each delay point
            delay_cm = []
            for i in range(0, true_y.shape[1]):
                delay_cm.append(confusion_matrix(y_true=true_y[:, i], y_pred=predict_y[:, i]))  # each confusion matrix in the list belongs to a delay point
            transition_cm[transition_type] = np.array(delay_cm)
        accuracy_bygroup.append(transition_accuracy)
        cm_bygroup.append(transition_cm)
    return accuracy_bygroup, cm_bygroup


##  get the accuracy and confusion matrix from all groups
def getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms, predict_window_shift_unit):
    # accuracy
    accuracy = {f'delay_{delay * feature_window_increment_ms * predict_window_shift_unit}_ms': value for delay, value in
        enumerate(overall_accuracy_with_delay.tolist())}

    # confusion matrix
    overall_cm = {}
    for delay in range(sum_cm_with_delay['transition_LW'].shape[0]):
        list_cm = [cm[delay, :, :] for label, cm in sum_cm_with_delay.items()]
        combined_cm = block_diag(*list_cm)
        cm_recall = np.around(combined_cm.astype('float') / combined_cm.sum(axis=1)[:, np.newaxis], 3)  # calculate cm recall
        overall_cm[f'delay_{delay * feature_window_increment_ms * predict_window_shift_unit}_ms'] = cm_recall

    return accuracy, overall_cm


##  save the model results to disk
def saveModelResults(subject, model_results, version, result_set):
    results = copy.deepcopy(model_results)
    for result in results:
        result['true_value'] = result['true_value'].values.tolist()
        result['predict_softmax'] = result['predict_softmax'].values.tolist()
        result['predict_value'] = result['predict_value'].tolist()

    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\model_results'
    result_file = f'subject_{subject}_Experiment_{version}_model_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    with open(result_path, 'w') as json_file:
        json.dump(results, json_file, indent=8)


##  read the model results from disk
def loadModelResults(subject, version, result_set):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\model_results'
    result_file = f'subject_{subject}_Experiment_{version}_model_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    # read json file
    with open(result_path) as json_file:
        result_json = json.load(json_file)
    for result in result_json:
        result['true_value'] = np.array(result['true_value'])
        result['predict_softmax'] = np.array(result['predict_softmax'])
        result['predict_value'] = np.array(result['predict_value'])

    return result_json


