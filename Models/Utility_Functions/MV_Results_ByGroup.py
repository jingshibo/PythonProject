'''
post processing using the majority vote method for multiple transition groups, and calculate the accuracy and confusion matrix.
'''

## import modules
from sklearn.metrics import confusion_matrix
from Models.Utility_Functions import Confusion_Matrix
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


## reorganize the non-grouped model classification results into different groups when prior information is used
def regroupModelResults(model_results):
    # convert numpy to pandas for easier grouping operation
    for result in model_results:
        result['true_value'] = pd.DataFrame(result['true_value'])
        result['predict_softmax'] = pd.DataFrame(result['predict_softmax'])
    # regroup the model results
    regrouped_results = []
    for result in model_results:
        transition_types = {}
        grouped_true_label = result['true_value'].groupby(0)  # group by categories (int value)
        categories = set(np.concatenate(result['true_value'].to_numpy()).tolist())
        # get grouped true value and predict value
        true_value_list = []
        for i in range(len(categories)):
            true_value_list.append(grouped_true_label.get_group(i))  # get the group of value i
        predict_prob_list = []
        for i in true_value_list:
            predict_prob_list.append((result['predict_softmax'].iloc[i.index.to_numpy().tolist(), :]))
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
        # transition_types['transition_SS'] = {
        #     'true_value': pd.concat([true_value_list[10], true_value_list[11], true_value_list[12], true_value_list[13]]).to_numpy(),
        #     'predict_softmax': pd.concat([predict_prob_list[10].iloc[:, 10:14], predict_prob_list[11].iloc[:, 10:14],
        #         predict_prob_list[12].iloc[:, 10:14], predict_prob_list[13].iloc[:, 10:14]]).to_numpy()}
        transition_types['transition_SS'] = {
            'true_value': pd.concat([true_value_list[10], true_value_list[11], true_value_list[12]]).to_numpy(),
            'predict_softmax': pd.concat([predict_prob_list[10].iloc[:, 10:13], predict_prob_list[11].iloc[:, 10:13],
                predict_prob_list[12].iloc[:, 10:13]]).to_numpy()}
        regrouped_results.append(transition_types)
    # convert softmax results to predicted values
    for result in regrouped_results:
        for transition_label, transition_results in result.items():
            transition_results['true_value'] = np.squeeze(transition_results['true_value'])  # remove one extra dimension
            transition_results['predict_value'] = np.argmax(transition_results['predict_softmax'], axis=-1) + transition_results['true_value'].min()  # return predicted labels
    return regrouped_results


## majority vote results
def majorityVoteResultsByGroup(group_results, window_per_repetition):
    # reunite the samples belonging to the same transition
    bin_results = []
    for each_group in group_results:
        bin_transitions = {}
        for transition_type, transition_results in each_group.items():
            true_y = []
            predict_y = []
            for key, value in transition_results.items():
                if key == 'true_value':
                    for i in range(0, len(value), window_per_repetition):
                        true_y.append(value[i: i+window_per_repetition])
                elif key == 'predict_value':
                    for i in range(0, len(value), window_per_repetition):
                        predict_y.append(value[i: i+window_per_repetition])
            bin_transitions[transition_type] = {"true_value": true_y, "predict_value": predict_y}
        bin_results.append(bin_transitions)

    # use majority vote to get a consensus result for each repetition
    majority_results = []
    for each_group in bin_results:
        majority_transitions = {}
        for transition_type, transition_results in each_group.items():
            true_y = []
            predict_y = []
            for key, value in transition_results.items():
                if key == 'true_value':
                    true_y = [np.bincount(i).argmax() for i in value]
                elif key == 'predict_value':
                    predict_y = [np.bincount(i).argmax() for i in value]
            majority_transitions[transition_type] = {"true_value": np.array(true_y), "predict_value": np.array(predict_y)}
        majority_results.append(majority_transitions)
    return majority_results

## calculate accuracy and cm values for each group
def getAccuracyPerGroup(majority_results):
    accuracy_bygroup = []
    cm_bygroup = []
    for each_group in majority_results:
        transition_cm = {}
        transition_accuracy = {}
        for transition_type, transition_result in each_group.items():
            true_y = transition_result['true_value']
            predict_y = transition_result['predict_value']
            numCorrect = np.count_nonzero(true_y == predict_y)
            transition_accuracy[transition_type] = numCorrect / len(true_y) * 100
            transition_cm[transition_type] = confusion_matrix(y_true=true_y, y_pred=predict_y)
        accuracy_bygroup.append(transition_accuracy)
        cm_bygroup.append(transition_cm)
    return accuracy_bygroup, cm_bygroup


## calculate average accuracy
def averageAccuracyByGroup(accuracy_bygroup, cm_bygroup):
    transition_groups = list(accuracy_bygroup[0].keys())  # list all transition types
    # average accuracy for each transition type
    average_accuracy = {transition: 0 for transition in transition_groups}  # initialize average accuracy list
    for group_values in accuracy_bygroup:
        for transition_type, transition_accuracy in group_values.items():
            average_accuracy[transition_type] = average_accuracy[transition_type] + transition_accuracy
    for transition_type, transition_accuracy in average_accuracy.items():
        average_accuracy[transition_type] = transition_accuracy / len(accuracy_bygroup)

    # overall accuracy for all transition types
    overall_accuracy = (average_accuracy['transition_LW'] * 1.5 + average_accuracy['transition_SA'] + average_accuracy['transition_SD'] +
                        average_accuracy['transition_SS']) / 4.5

    # overall cm among groups
    sum_cm = {transition: 0 for transition in transition_groups}   # initialize overall cm list
    for group_values in cm_bygroup:
        for transition_type, transition_cm in group_values.items():
            sum_cm[transition_type] = sum_cm[transition_type] + transition_cm

    return average_accuracy, overall_accuracy, sum_cm


## plot confusion matrix
def confusionMatrix(sum_cm, is_recall=False):
    # create a diagonal matrix from multiple arrays.
    list_cm = [cm for label, cm in sum_cm.items()]
    overall_cm = block_diag(*list_cm)

    # the label order in the classes list should correspond to the one hot labels, which is a alphabetical order
    # class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'SALW', 'SASA', 'SASS', 'SDLW', 'SDSD', 'SDSS', 'SSLW', 'SSSA', 'SSSD', 'SSSS']
    class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'SALW', 'SASA', 'SASS', 'SDLW', 'SDSD', 'SDSS', 'SSLW', 'SSSA', 'SSSD']
    plt.figure()
    cm_recall = Confusion_Matrix.plotConfusionMatrix(overall_cm, class_labels, normalize=is_recall)
    return cm_recall