'''
reorganize the classification results
'''


## import modules
import pandas as pd
import numpy as np
from collections import defaultdict
import copy


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
            # transition_types['transition_SS'] = {
            #     'true_value': pd.concat([true_value_list[10], true_value_list[11], true_value_list[12], true_value_list[13]]).to_numpy(),
            #     'predict_softmax': pd.concat([predict_prob_list[10].iloc[:, 10:14], predict_prob_list[11].iloc[:, 10:14],
            #         predict_prob_list[12].iloc[:, 10:14], predict_prob_list[13].iloc[:, 10:14]]).to_numpy()}
            transition_types['transition_SS'] = {
                'true_value': pd.concat([true_value_list[10], true_value_list[11], true_value_list[12]]).to_numpy(),
                'predict_softmax': pd.concat([predict_prob_list[10].iloc[:, 10:13], predict_prob_list[11].iloc[:, 10:13],
                    predict_prob_list[12].iloc[:, 10:13]]).to_numpy()}
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


## reduce the number of test set prediction results: only keep the results after the initial_predict_time
def reducePredictResults(reorganized_softmax, reorganized_prediction, initial_predict_time):
    reduced_softmax = copy.deepcopy(reorganized_softmax)
    for group_number, group_value in enumerate(reorganized_softmax):
        for transition_type, transition_value in group_value.items():
            reduced_softmax[group_number][transition_type] = transition_value[:, initial_predict_time:, :]
    reduced_prediction = copy.deepcopy(reorganized_prediction)
    for group_number, group_value in enumerate(reorganized_prediction):
        for transition_type, transition_value in group_value.items():
            reduced_prediction[group_number][transition_type] = transition_value[initial_predict_time:, :]

    return reduced_softmax, reduced_prediction


## find the first timestamps at which the softmax value is larger than the threshold
def findFirstTimestamp(reduced_softmax, threshold=0.99):
    first_timestamps_above_threshold = copy.deepcopy(reduced_softmax)
    for group_value in first_timestamps_above_threshold:
        for transition_type, transition_value in group_value.items():
            #  for each repetition, find the first timestamp at which the softmax value is larger than the threshold
            largest_softmax = np.transpose(np.amax(transition_value, axis=2))  # for each timestamp, return the largest softmax value among all categories
            first_timestamp = np.argmax(largest_softmax > threshold, axis=0)  # for each repetition, return the first timestamp index where the softmax value is above the threshold

            # dedicated addressing the special case when a repetition has all the softmax values (from all timestamps) below the threshold
            first_softmax = largest_softmax[first_timestamp.tolist(), list(range(largest_softmax.shape[1]))]  # find the first softmax value above the threshold for each repetition
            threshold_test = first_softmax / threshold  # calculate the ratio to show if there are any repetitions with all the softmax
            # values below the threshold
            repetition_low_softmax = np.transpose(np.squeeze(np.argwhere(threshold_test < 1)))  # return the index of the repetition with all softmax values below the threshold
            first_timestamp[repetition_low_softmax.tolist()] = -1  # change the timestamp index  to -1, for the repetition with all softmax values below the threshold

            group_value[transition_type] = first_timestamp

    return first_timestamps_above_threshold


## query the prediction based on timestamps from the reorganized_prediction table
def getSlidingPredictResults(reduced_prediction, first_timestamps, initial_predict_time, shift_unit, increment_ms):
    delay_unit_ms = increment_ms * shift_unit  # the window increment value(ms) * the shift number = the delay for each timestamp
    sliding_prediction = copy.deepcopy(reduced_prediction)
    for group_number, group_value in enumerate(first_timestamps):
        for transition_type, transition_value in group_value.items():
            # get the prediction results based on the timestamp information
            predict_result = sliding_prediction[group_number][transition_type][
                transition_value.tolist(), list(range(transition_value.shape[0]))]
            # convert timestamp index to delay
            delay_ms = (transition_value + initial_predict_time) * delay_unit_ms
            # predict category AND delay value (the first row is prediction results, the second row is the delay value)
            sliding_prediction[group_number][transition_type] = np.array([predict_result, delay_ms])

            # dedicated addressing the special case when the first_timestamps is negative (due to the -1 timestamp value assigned)
            negative_indices = np.argwhere(sliding_prediction[group_number][transition_type][1, :] < initial_predict_time * delay_unit_ms)
            # negative delay value means there is no results with probabilities higher than the threshold in this repetition
            if negative_indices.size > 0:
                for column in np.nditer(negative_indices):
                    # use the most common prediction as the result
                    prediction = np.bincount(reduced_prediction[group_number][transition_type][:, column]).argmax()
                    # use the maximum delay value instead
                    delay_ms = delay_unit_ms * (reduced_prediction[group_number][transition_type].shape[0] + initial_predict_time - 1)
                    sliding_prediction[group_number][transition_type][0, column] = prediction
                    sliding_prediction[group_number][transition_type][1, column] = delay_ms

    return sliding_prediction

