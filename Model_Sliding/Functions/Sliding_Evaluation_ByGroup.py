'''
evaluating the classification results (accuracy AND delay)
'''

##
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import copy


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
    integrated_results = copy.deepcopy(reorganized_truevalues)
    for group_number, group_value in enumerate(sliding_prediction):
        for transition_type, transition_value in group_value.items():
            # put true and predict information into a single array (order: predict_category, true_category, category_difference, delay_value)
            combined_array = np.array([transition_value[0, :], reorganized_truevalues[group_number][transition_type],
                reorganized_truevalues[group_number][transition_type] - transition_value[0, :], transition_value[1, :]])
            unequal_columns = np.where(combined_array[2, :] != 0)[0]  # the column index where the predict values differ from the true values

            # remove the columns with the true values and predict values that are unequal
            if unequal_columns.size > 0:
                integrated_results[group_number][transition_type] = np.delete(combined_array, unequal_columns, 1)
            else:
                integrated_results[group_number][transition_type] = combined_array

    return integrated_results


##  calculate the count, mean and std values of delay for each category
def countDelay(integrated_results):
    combined_results = np.array([]).reshape(2, 0)
    for group_number, group_value in enumerate(integrated_results):
        for transition_type, transition_value in group_value.items():
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
    # the percentage of delay values for each category
    category_count = count_delay_category.groupby(['Category'])['Count'].sum()
    category_ratio = pd.Series(dtype='float64')
    for category in range(len(category_count)):
        ratio = count_delay_category.loc[count_delay_category['Category'] == category, 'Count'] / category_count[category]
        category_ratio = pd.concat([category_ratio, ratio])
    count_delay_category['Percentage'] = category_ratio

    # sum up the count of delay for all categories
    unique_delay = np.sort(count_delay_category["Delay(ms)"].unique())
    delay = {}
    for delay_value in unique_delay.tolist():
        delay[delay_value] = count_delay_category.loc[count_delay_category['Delay(ms)'] == delay_value, 'Count'].sum()
    count_delay_overall = pd.DataFrame(delay.items(), columns=['Delay(ms)', 'Count'])
    # the percentage of delay for all categories
    count_delay_overall['Percentage'] = count_delay_overall['Count'] / count_delay_overall['Count'].sum()

    return count_delay_category, count_delay_overall, mean_delay, std_delay



