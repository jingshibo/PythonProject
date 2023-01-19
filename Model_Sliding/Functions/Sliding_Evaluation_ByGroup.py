'''
evaluating the classification results (accuracy AND delay)
'''

## import modules
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.linalg import block_diag
import copy
from Models.Utility_Functions import MV_Results_ByGroup
from Model_Sliding.Functions import Sliding_Results_ByGroup


## calculate accuracy and cm values for each group
def getAccuracyPerGroup(sliding_prediction, reorganized_truevalues):
    accuracy = []
    cm = []
    for group_number, group_value in enumerate(sliding_prediction):
        transition_cm = {}
        transition_accuracy = {}
        for transition_type, transition_result in group_value.items():
            true_y = reorganized_truevalues[group_number][transition_type]
            predict_y = transition_result[0, :]
            numCorrect = np.count_nonzero(true_y == predict_y)
            transition_accuracy[transition_type] = numCorrect / len(true_y) * 100
            transition_cm[transition_type] = confusion_matrix(y_true=true_y, y_pred=predict_y)
        accuracy.append(transition_accuracy)
        cm.append(transition_cm)
    return accuracy, cm


##  combine the true and predict value into a single array for later processing
def integrateResults(sliding_prediction, reorganized_truevalues):
    correct_results = copy.deepcopy(reorganized_truevalues)  # integrate the correct result information into one array
    false_results = copy.deepcopy(reorganized_truevalues)  # integrate the false results information into one array
    for group_number, group_value in enumerate(sliding_prediction):
        for transition_type, transition_value in group_value.items():
            # put true and predict values into a single array (order: predict_category, true_category, category_difference, delay_value)
            combined_array = np.array([transition_value[0, :], reorganized_truevalues[group_number][transition_type],
                reorganized_truevalues[group_number][transition_type] - transition_value[0, :], transition_value[1, :]])
            unequal_columns = np.where(combined_array[2, :] != 0)[0]  # the column index where the predict values differ from the true values

            # remove the columns with the true values and predict values that are unequal
            if unequal_columns.size > 0:
                correct_results[group_number][transition_type] = np.delete(combined_array, unequal_columns, 1)
                false_results[group_number][transition_type] = combined_array[:, unequal_columns]
            else:
                correct_results[group_number][transition_type] = combined_array
                false_results[group_number][transition_type] = np.array([])

    return correct_results, false_results


##  calculate the count, mean and std values of delay for each category
def countDelay(integrated_results):
    combined_results = np.array([]).reshape(2, 0)
    for group_number, group_value in enumerate(integrated_results):
        for transition_type, transition_value in group_value.items():
            if transition_value.size != 0:
                transition_value = np.delete(transition_value, [0, 2], 0)  # only keep the true_value row and delay_value row
                combined_results = np.hstack((combined_results, transition_value))  # combine all results into one array

    # mean and std value of delay for each category
    regrouped_delay = pd.DataFrame(combined_results).T.groupby([0])  # group delay values by categories
    mean_delay = regrouped_delay.mean()
    std_delay = regrouped_delay.std()

    # the count of delay values for each category
    regrouped_delay = pd.DataFrame(combined_results).T.groupby([0, 1])  # group delay values by categories and delays
    count_delay_category = regrouped_delay.size().reset_index(name="count")
    count_delay_category.columns = ['Category', 'Delay(ms)', 'Count']
    count_delay_category['Category'] = count_delay_category['Category'].astype('int64')  # convert float category value to int

    # the percentage of delay values for each category
    category_count = count_delay_category.groupby(['Category'])['Count'].sum()
    category_ratio = pd.Series(dtype='float64')
    for category in category_count.index.tolist():
        ratio = count_delay_category.loc[count_delay_category['Category'] == category, 'Count'] / category_count[category]
        category_ratio = pd.concat([category_ratio, ratio])
    count_delay_category['Percentage'] = category_ratio
    count_delay_category['Category'] = count_delay_category['Category'].replace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'SALW', 'SASA', 'SASS', 'SDLW', 'SDSD', 'SDSS', 'SSLW', 'SSSA', 'SSSD'])  # replace number by name

    # sum up the count of delay for all categories
    unique_delay = np.sort(count_delay_category["Delay(ms)"].unique())
    delay = {}
    for delay_value in unique_delay.tolist():
        delay[delay_value] = count_delay_category.loc[count_delay_category['Delay(ms)'] == delay_value, 'Count'].sum()
    count_delay_overall = pd.DataFrame(delay.items(), columns=['Delay(ms)', 'Count'])
    # the percentage of correct predicted results at this delay timepoint for all categories
    count_delay_overall['Percentage'] = count_delay_overall['Count'] / count_delay_overall['Count'].sum()
    # # accumulate the percentage value of the previous rows
    count_delay_overall['Percentage_cum'] = count_delay_overall['Percentage'].values.cumsum()
    count_delay_overall.insert(3, 'Percentage_cum', count_delay_overall.pop('Percentage_cum'))  # change the column order

    return count_delay_category, count_delay_overall, mean_delay, std_delay


## calculate the prediction accuracy at each delay timestamp point
def delayAccuracy(predict_delay_overall, false_delay_overall):
    # accurate rate of the prediction decisions made at each delay timestamp
    correct_prediction = []
    count_all = []
    for index, row in predict_delay_overall.iterrows():
        if row['Delay(ms)'] in false_delay_overall['Delay(ms)'].values:
            false_count = false_delay_overall.loc[false_delay_overall['Delay(ms)'] == row['Delay(ms)'], 'Count'].to_numpy()[0]
            correct_count = predict_delay_overall.loc[index, 'Count']
            total_number = false_count + correct_count
            correct_ratio = correct_count / total_number
        else:
            correct_ratio = 1.0
            total_number = predict_delay_overall.loc[index, 'Count']
        correct_prediction.append(correct_ratio)
        count_all.append(total_number)
    predict_delay_overall['Accuracy'] = correct_prediction
    predict_delay_overall['Count_all'] = count_all
    predict_delay_overall.insert(2, 'Count_all', predict_delay_overall.pop('Count_all'))  # change the column order
    predict_delay_overall.insert(3, 'Accuracy', predict_delay_overall.pop('Accuracy'))  # change the column order

    # error rate of the prediction decisions made at each delay timestamp
    false_prediction = []
    count_all = []
    for index, row in false_delay_overall.iterrows():
        if row['Delay(ms)'] in predict_delay_overall['Delay(ms)'].values:
            correct_count = predict_delay_overall.loc[predict_delay_overall['Delay(ms)'] == row['Delay(ms)'], 'Count'].to_numpy()[0]
            false_count = false_delay_overall.loc[index, 'Count']
            total_number = false_count + correct_count
            error_ratio = false_count / total_number
        else:
            error_ratio = 1.0
            total_number = false_delay_overall.loc[index, 'Count']
        false_prediction.append(error_ratio)
        count_all.append(total_number)
    false_delay_overall['Error_rate'] = false_prediction
    false_delay_overall['Count_all'] = count_all
    false_delay_overall.insert(2, 'Count_all', false_delay_overall.pop('Count_all'))  # change the column order
    false_delay_overall.insert(3, 'Error_rate', false_delay_overall.pop('Error_rate'))  # change the column order
    a=1
    return predict_delay_overall, false_delay_overall


##  group the classification results together from diffferent initial_predict_time settings
def groupSlidingResults(model_results, shift_unit, increment_ms, end_predict_time=-1, threshold=0.999):
    # reorganize the results
    # regroup the model results using prior information (no majority vote used, what we need here is the grouped accuracy calculation)
    regrouped_results = Sliding_Results_ByGroup.regroupModelResults(model_results)
    # reorganize the regrouped model results based on the timestamps
    reorganized_softmax, reorganized_prediction, reorganized_truevalues = Sliding_Results_ByGroup.reorganizePredictValues(regrouped_results)

    group_sliding_results = {}
    shift_total_number = reorganized_prediction[0]['transition_LW'].shape[0]  # the total number of sliding for classfication
    for initial_predict_time in range(shift_total_number):
        if initial_predict_time < end_predict_time:  # only applies for those when initial_predict_time is smaller than end_predict_time
            # preprocessing the results
            # only keep the results after the initial_predict_time
            reduced_softmax, reduced_prediction = Sliding_Results_ByGroup.reducePredictResults(reorganized_softmax, reorganized_prediction, initial_predict_time, end_predict_time)
            #  find the first timestamps at which the softmax value is larger than the threshold
            first_timestamps = Sliding_Results_ByGroup.findFirstTimestamp(reduced_softmax, threshold)
            # get the predict results based on timestamps from the reorganized_prediction table and convert the timestamp to delay(ms)
            sliding_prediction = Sliding_Results_ByGroup.getSlidingPredictResults(reduced_prediction, first_timestamps, initial_predict_time, shift_unit, increment_ms)

            # evaluate the prediction results
            # calculate the prediction accuracy
            accuracy_bygroup, cm_bygroup = getAccuracyPerGroup(sliding_prediction, reorganized_truevalues)
            average_accuracy, overall_accuracy, sum_cm = MV_Results_ByGroup.averageAccuracy(accuracy_bygroup, cm_bygroup)
            list_cm = [cm for label, cm in sum_cm.items()]
            overall_cm = block_diag(*list_cm)
            cm_recall = np.around(overall_cm.astype('float') / overall_cm.sum(axis=1)[:, np.newaxis], 3)  # calculate cm recall

            # calculate the prediction delay (separately for those with correct or false prediction results)
            correct_results, false_results = integrateResults(sliding_prediction, reorganized_truevalues)
            predict_delay_category, predict_delay_overall, mean_delay, std_delay = countDelay(correct_results)
            false_delay_category, false_delay_overall, false_delay_mean, false_delay_std = countDelay(false_results)
            predict_delay_overall, false_delay_overall = delayAccuracy(predict_delay_overall, false_delay_overall)

            # group the prediction results
            group_result = {'average_accuracy': average_accuracy, 'overall_accuracy': overall_accuracy, 'cm_recall': cm_recall,
                'predict_delay_category': predict_delay_category, 'predict_delay_overall': predict_delay_overall,
                'false_delay_category': false_delay_category, 'false_delay_overall': false_delay_overall}

            group_sliding_results[f'initial_{increment_ms * shift_unit * initial_predict_time}_ms'] = group_result

    return group_sliding_results

