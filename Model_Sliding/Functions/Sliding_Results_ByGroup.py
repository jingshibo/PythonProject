## import modules
from sklearn.metrics import confusion_matrix
from Models.Utility_Functions import Confusion_Matrix
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from collections import defaultdict


## reorganize the non-grouped model classification results into different groups when prior information is used
def regroupModelResults(model_results):

    # convert numpy to pandas for easier grouping operation
    for result in model_results:
        for shift_number, shift_value in result.items():
            shift_value['true_value'] = pd.DataFrame(shift_value['true_value'])
            shift_value['predict_softmax'] = pd.DataFrame(shift_value['predict_softmax'])

    # regroup the model results
    regrouped_results = []
    for result in model_results:
        regrouped_shift = {}
        for shift_number, shift_value in result.items():
            transition_types = {}
            grouped_true_label = shift_value['true_value'].groupby(0)  # group by categories (int value)
            categories = set(np.concatenate(shift_value['true_value'].to_numpy()).tolist())
            # get grouped true value and predict value
            true_value_list = []
            for i in range(len(categories)):
                true_value_list.append(grouped_true_label.get_group(i))  # get the group of value i
            predict_prob_list = []
            for i in true_value_list:
                predict_prob_list.append((shift_value['predict_softmax'].iloc[i.index.to_numpy().tolist(), :]))
            # reorganize true value and predict value
            transition_types['transition_LW'] = {
                'true_value': pd.concat([true_value_list[0], true_value_list[1], true_value_list[2], true_value_list[3]]).to_numpy(),
                'predict_softmax': pd.concat([predict_prob_list[0].iloc[:, 0:4], predict_prob_list[1].iloc[:, 0:4],
                    predict_prob_list[2].iloc[:, 0:4], predict_prob_list[3].iloc[:, 0:4]]).to_numpy()}
            transition_types['transition_SA'] = {
                'true_value': pd.concat([true_value_list[4], true_value_list[5], true_value_list[6]]).to_numpy(), 'predict_softmax': pd.concat(
                    [predict_prob_list[4].iloc[:, 4:7], predict_prob_list[5].iloc[:, 4:7], predict_prob_list[6].iloc[:, 4:7]]).to_numpy()}
            transition_types['transition_SD'] = {
                'true_value': pd.concat([true_value_list[7], true_value_list[8], true_value_list[9]]).to_numpy(), 'predict_softmax': pd.concat(
                    [predict_prob_list[7].iloc[:, 7:10], predict_prob_list[8].iloc[:, 7:10], predict_prob_list[9].iloc[:, 7:10]]).to_numpy()}
            transition_types['transition_SS'] = {
                'true_value': pd.concat([true_value_list[10], true_value_list[11], true_value_list[12], true_value_list[13]]).to_numpy(),
                'predict_softmax': pd.concat([predict_prob_list[10].iloc[:, 10:14], predict_prob_list[11].iloc[:, 10:14],
                    predict_prob_list[12].iloc[:, 10:14], predict_prob_list[13].iloc[:, 10:14]]).to_numpy()}
            # transition_types['transition_LW'] = {
            #     'true_value': pd.concat([true_value_list[0], true_value_list[1], true_value_list[2], true_value_list[3]]).to_numpy(),
            #     'predict_softmax': pd.concat([predict_prob_list[0].iloc[:, 0:4], predict_prob_list[1].iloc[:, 0:4],
            #         predict_prob_list[2].iloc[:, 0:4], predict_prob_list[3].iloc[:, 0:4]]).to_numpy()}
            # transition_types['transition_SA'] = {
            #     'true_value': pd.concat([true_value_list[4], true_value_list[5], true_value_list[6]]).to_numpy(), 'predict_softmax': pd.concat(
            #         [predict_prob_list[4].iloc[:, 4:7], predict_prob_list[5].iloc[:, 4:7], predict_prob_list[6].iloc[:, 4:7]]).to_numpy()}
            # transition_types['transition_SD'] = {
            #     'true_value': pd.concat([true_value_list[7], true_value_list[8], true_value_list[9]]).to_numpy(), 'predict_softmax': pd.concat(
            #         [predict_prob_list[7].iloc[:, 7:10], predict_prob_list[8].iloc[:, 7:10], predict_prob_list[9].iloc[:, 7:10]]).to_numpy()}
            # transition_types['transition_SS'] = {
            #     'true_value': pd.concat([true_value_list[10], true_value_list[11], true_value_list[12]]).to_numpy(),
            #     'predict_softmax': pd.concat([predict_prob_list[10].iloc[:, 10:13], predict_prob_list[11].iloc[:, 10:13],
            #         predict_prob_list[12].iloc[:, 10:13]]).to_numpy()}
            regrouped_shift[shift_number] = transition_types

        regrouped_results.append(regrouped_shift)

    # convert softmax results to predicted values
    for result in regrouped_results:
        for shift_number, shift_value in result.items():
            for transition_label, transition_results in shift_value.items():
                transition_results['true_value'] = np.squeeze(transition_results['true_value'])  # remove one extra dimension
                transition_results['predict_value'] = np.argmax(transition_results['predict_softmax'], axis=-1) + transition_results['true_value'].min()  # return predicted labels
    return regrouped_results


## reorganize softmax results based on the timestamps
def reorganizePredictValues(regrouped_results):
    # put softmax values and predict_results (from different timestamps) together, into the corresponding transition type
    # and reorganize true values in order to match the structure of predicted values

    softmax_values = []
    predict_results = []
    true_results = []
    for each_group in regrouped_results:
        softmax_transition = defaultdict(list)
        predict_transition = defaultdict(list)
        true_transition = {}
        for shift_number, shift_value in each_group.items():
            for transition_type, transition_value in shift_value.items():
                softmax_transition[transition_type].append(transition_value['predict_softmax'])
                predict_transition[transition_type].append(transition_value['predict_value'])
                true_transition[transition_type] = transition_value['true_value']
        softmax_values.append(softmax_transition)  # the softmax value (of all categories) for each timestamps in each repetition
        predict_results.append(predict_transition)  # the predict result for each timestamps in each repetition
        true_results.append(true_transition)  # the true value for each repetition

    # convert softmax list into numpy array (structure: [samples, timestamp, categories])
    softmax_reorganized = []
    for each_group in softmax_values:
        softmax = {}
        for transition_type, transition_value in each_group.items():
            if transition_type != 'default_factory':
                softmax[transition_type] = np.stack(transition_value, axis=1)
        softmax_reorganized.append(softmax)

    # convert predict list into numpy array (structure: [timestamp, samples])
    predict_reorganized = []
    for each_group in predict_results:
        predict = {}
        for transition_type, transition_value in each_group.items():
            if transition_type != 'default_factory':
                predict[transition_type] = np.transpose(np.stack(transition_value, axis=1))
        predict_reorganized.append(predict)


    return softmax_reorganized, predict_reorganized, true_results


## calculate accuracy and cm values for each group
def getAccuracyPerGroup(regrouped_results):
    accuracy = []
    cm = []
    for each_group in regrouped_results:
        shift_accuracy = {}
        shift_cm = {}
        for shift_number, shift_value in each_group.items():
            transition_accuracy = {}
            transition_cm = {}
            for transition_type, transition_result in shift_value.items():
                true_y = transition_result['true_value']
                predict_y = transition_result['predict_value']
                numCorrect = np.count_nonzero(true_y == predict_y)
                transition_accuracy[transition_type] = numCorrect / len(true_y) * 100
                transition_cm[transition_type] = confusion_matrix(y_true=true_y, y_pred=predict_y)
            shift_accuracy[shift_number] = transition_accuracy
            shift_cm[shift_number] = transition_cm
        accuracy.append(shift_accuracy)
        cm.append(shift_cm)
    return accuracy, cm

