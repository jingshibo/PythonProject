'''
evaluating the classification results (accuracy AND delay)
'''

## import modules
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.linalg import block_diag
import copy
from Transition_Prediction.Models.Utility_Functions import MV_Results_ByGroup
from Transition_Prediction.Model_Sliding.GRU.Functions import Sliding_Results_ByGroup


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
    count_delay_category = regrouped_delay.size().reset_index(name="Count")
    count_delay_category.columns = ['Category', 'Delay(ms)', 'Count']
    count_delay_category['Category'] = count_delay_category['Category'].astype('int64')  # convert float category value to int

    # the percentage of delay values for each category
    category_count = count_delay_category.groupby(['Category'])['Count'].sum()
    category_percentage = pd.Series(dtype='float64')
    category_percentage_cum = pd.Series(dtype='float64')
    for category in category_count.index.tolist():
        percentage = count_delay_category.loc[count_delay_category['Category'] == category, 'Count'] / category_count[category]
        percentage_cum = pd.Series(percentage.values.cumsum())
        category_percentage = pd.concat([category_percentage, percentage])
        category_percentage_cum = pd.concat([category_percentage_cum, percentage_cum])
    count_delay_category['Percentage'] = category_percentage.to_numpy()
    count_delay_category['Percentage_cum'] = category_percentage_cum.to_numpy()
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
def delayAccuracy(predict_delay_overall, predict_delay_category, false_delay_overall, false_delay_category):
    # define the function to calculate the accurate rate of the overall prediction decisions made at each delay timestamp
    def calculateOverallAccuracy(target_data, counterpart_data):
        ratio = []
        count_all = []
        for index, row in target_data.iterrows():  # loop over each row
            if row['Delay(ms)'] in counterpart_data['Delay(ms)'].values:
                counterpart_count = counterpart_data.loc[counterpart_data['Delay(ms)'] == row['Delay(ms)'], 'Count'].to_numpy()[0]
                target_count = target_data.loc[index, 'Count']
                total_number = counterpart_count + target_count
                target_ratio = target_count / total_number
            else:
                target_ratio = 1.0
                total_number = target_data.loc[index, 'Count']
            ratio.append(target_ratio)
            count_all.append(total_number)
        target_data['Accuracy'] = ratio
        target_data['Count_all'] = count_all
        target_data.insert(1, 'Count_all', target_data.pop('Count_all'))  # change the column order
        target_data.insert(3, 'Accuracy', target_data.pop('Accuracy'))  # change the column order

        return target_data
    # call the function defined above
    predict_delay_overall = calculateOverallAccuracy(predict_delay_overall, false_delay_overall)
    false_delay_overall = calculateOverallAccuracy(false_delay_overall, predict_delay_overall)
    predict_delay_overall.columns = ['Delay(ms)', 'Predict_Number', 'Correct_Number', 'Accuracy', 'Correct_Percentage', 'Percentage_Cum']
    predict_delay_overall.columns = ['Delay(ms)', 'Predict_Number', 'False_Number', 'Error_Rate', 'False_Percentage', 'Percentage_Cum']

    # define the function to calculate the accurate rate of the category prediction decisions made at each delay timestamp
    def calculateCategoryAccuracy(target_data, counterpart_data):
        ratio = []
        count_all = []
        for index, row in target_data.iterrows():  # loop over each row
            if row['Category'] in counterpart_data['Category'].values:  # if Category value and Delay(ms) value exist in counterpart_data
                if row['Delay(ms)'] in counterpart_data.loc[counterpart_data['Category'] == row['Category'], 'Delay(ms)'].values:
                    counterpart_count = counterpart_data.loc[(counterpart_data['Delay(ms)'] == row['Delay(ms)']) & (
                            counterpart_data['Category'] == row['Category']), 'Count'].to_numpy()[0]
                    target_count = target_data.loc[index, 'Count']
                    total_number = counterpart_count + target_count
                    target_ratio = target_count / total_number
                else:
                    target_ratio = 1.0
                    total_number = target_data.loc[index, 'Count']
            else:
                target_ratio = 1.0
                total_number = target_data.loc[index, 'Count']
            ratio.append(target_ratio)
            count_all.append(total_number)
        target_data['Accuracy'] = ratio
        target_data['Count_all'] = count_all
        target_data.insert(2, 'Count_all', target_data.pop('Count_all'))  # change the column order
        target_data.insert(4, 'Accuracy', target_data.pop('Accuracy'))  # change the column order

        return target_data
    # call the function defined above
    predict_delay_category = calculateCategoryAccuracy(predict_delay_category, false_delay_category)
    false_delay_category = calculateCategoryAccuracy(false_delay_category, predict_delay_category)
    predict_delay_category.columns = ['Category', 'Delay(ms)', 'Predict_Number', 'Correct_Number', 'Accuracy', 'Correct_Percentage', 'Percentage_Cum']
    false_delay_category.columns = ['Category', 'Delay(ms)', 'Predict_Number', 'False_Number', 'Error_Rate', 'False_Percentage', 'Percentage_Cum']

    return predict_delay_overall, predict_delay_category, false_delay_overall, false_delay_category


##  group the classification results together from diffferent initial_predict_time settings
def groupSlidingResults(model_results, predict_window_shift_unit, feature_window_increment_ms, end_predict_time, threshold=0.999):
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
            reduced_softmax, reduced_prediction = Sliding_Results_ByGroup.reducePredictResults(reorganized_softmax, reorganized_prediction,
                initial_predict_time, end_predict_time)
            #  find the first timestamps at which the softmax value is larger than the threshold
            first_timestamps = Sliding_Results_ByGroup.findFirstTimestamp(reduced_softmax, threshold)
            # get the predict results based on timestamps from the reorganized_prediction table and convert the timestamp to delay(ms)
            sliding_prediction = Sliding_Results_ByGroup.getSlidingPredictResults(reduced_prediction, first_timestamps,
                initial_predict_time, predict_window_shift_unit, feature_window_increment_ms)

            # calculate the prediction accuracy and confusion matrix
            accuracy_bygroup, cm_bygroup = getAccuracyPerGroup(sliding_prediction, reorganized_truevalues)
            average_accuracy, overall_accuracy, sum_cm = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup, cm_bygroup)
            list_cm = [cm for label, cm in sum_cm.items()]
            overall_cm = block_diag(*list_cm)
            cm_recall = np.around(overall_cm.astype('float') / overall_cm.sum(axis=1)[:, np.newaxis], 3)  # calculate cm recall

            # calculate the prediction delay (separately for those with correct or false prediction results)
            correct_results, false_results = integrateResults(sliding_prediction, reorganized_truevalues)
            predict_delay_category, predict_delay_overall, mean_delay, std_delay = countDelay(correct_results)
            false_delay_category, false_delay_overall, false_delay_mean, false_delay_std = countDelay(false_results)
            predict_delay_overall, predict_delay_category, false_delay_overall, false_delay_category = delayAccuracy(predict_delay_overall,
                predict_delay_category, false_delay_overall, false_delay_category)

            # group the prediction results
            group_result = {'average_accuracy': average_accuracy, 'overall_accuracy': overall_accuracy, 'cm_recall': cm_recall,
                'predict_delay_category': predict_delay_category, 'predict_delay_overall': predict_delay_overall,
                'false_delay_category': false_delay_category, 'false_delay_overall': false_delay_overall}

            group_sliding_results[f'initial_{feature_window_increment_ms * predict_window_shift_unit * initial_predict_time}_ms'] = group_result

    return group_sliding_results


## get the classification accuracy at each delay position
def getResultsEachDelay(model_results, predict_window_shift_unit, feature_window_increment_ms):
    results = {}
    predict_window_increment = predict_window_shift_unit * feature_window_increment_ms

    # read the prediction value at the each delay position from the sliding results
    for delay in range(0, 513, predict_window_increment):  # loop over each delay position to get the prediction result
        end_predict_timestamp = int(delay / predict_window_increment) + 1  # define the end prediction timestamp at which the predict ends
        group_sliding_results = groupSlidingResults(model_results, predict_window_shift_unit, feature_window_increment_ms, end_predict_timestamp)

        # only retain the last sliding result, which is the prediction result at the given maximum delay time point
        last_window_value = list(group_sliding_results.items())[-1]
        delay_value = "delay_" + last_window_value[0][8:]  # modify the key name string
        results[delay_value] = last_window_value[1]

    return results
