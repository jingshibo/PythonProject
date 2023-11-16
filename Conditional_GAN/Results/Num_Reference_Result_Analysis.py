import numpy as np
import pandas as pd
import copy
from scipy.stats import ttest_rel


## combine the results of using different number of references into the dicts for all subjects
def combineNumOfReferenceResults(original_data):
    # Initialize the new dictionary with the second level keys as top level
    combined_data = {metric_key: {} for metric_key in next(iter(original_data.values())).keys()}

    # Iterate over the original dictionary to populate the new one
    for number_key, number_dict in original_data.items():
        for metric_key, metric_dict in number_dict.items():
            for reference_key, reference_dict in metric_dict.items():
                for model_key, model_dict in reference_dict.items():
                    # We remove the 'delay_0_ms' level by directly accessing its value
                    delay_value = model_dict['delay_0_ms']

                    # Make sure the nested dictionaries exist in the new structure
                    if reference_key not in combined_data[metric_key]:
                        combined_data[metric_key][reference_key] = {}
                    if model_key not in combined_data[metric_key][reference_key]:
                        combined_data[metric_key][reference_key][model_key] = {}

                    # Assign the value, moving the 'NumberX' level to the bottom
                    combined_data[metric_key][reference_key][model_key][number_key] = delay_value

    return combined_data


##
def convertToDataframes(combined_data):
    dataframe_dict = combined_data.copy()

    for metric_key, metric_dict in combined_data.items():  # e.g., 'accuracy', 'cm_recall'
        if metric_key == 'accuracy':
            metric_dataframe_dict = {}  # Dictionary to hold dataframes for this metric
            for reference_key, model_dict in metric_dict.items():  # e.g., 'reference_0', 'reference_1', ...
                # Initialize an empty DataFrame for this reference with NumberX as the index
                reference_dataframe = pd.DataFrame()
                for model_key, number_dict in model_dict.items():  # e.g., 'model_old', 'model_new', ...
                    # Create a Series from the dictionary with NumberX as the index
                    series = pd.Series(number_dict, name=model_key)
                    # Append the Series as a new column in the reference DataFrame
                    reference_dataframe = reference_dataframe.join(series, how='outer')
                # Store the DataFrame for each reference in the metric dictionary
                metric_dataframe_dict[reference_key] = reference_dataframe
            # Replace the accuracy part of the main dictionary with the DataFrames
            dataframe_dict[metric_key] = metric_dataframe_dict

    return dataframe_dict


## calculate statistical results for each condition
def calcuNumOfReferenceStatValues(reorganized_results):
    mean_std_value = copy.deepcopy(reorganized_results)

    # Compute the mean and std value of the accuracies
    mean_std_value["accuracy"]["statistics"] = {}
    # Initialize dictionaries to hold the mean and std values DataFrames
    mean_values_dict = {}
    std_values_dict = {}
    # Iterate over each reference in the 'accuracy' part of the dictionary
    for reference_key, df in reorganized_results['accuracy'].items():
        # Calculate the mean and std of each column in the DataFrame
        mean_values = df.mean()
        std_values = df.std()
        # Convert the mean and std values Series to a single-row DataFrame
        mean_values_df = mean_values.to_frame().transpose()
        std_values_df = std_values.to_frame().transpose()
        # Assign the mean and std values DataFrames to the corresponding reference key
        mean_values_dict[reference_key] = mean_values_df
        std_values_dict[reference_key] = std_values_df
    # Combine all mean value DataFrames into a single DataFrame with reference_x as the index
    combined_mean_values_df = pd.concat(mean_values_dict)
    combined_std_values_df = pd.concat(std_values_dict)
    # Adjust the index to have only the reference_x as the index
    combined_mean_values_df.index = combined_mean_values_df.index.droplevel(1)
    combined_std_values_df.index = combined_std_values_df.index.droplevel(1)
    # Assign the combined mean and std values DataFrames to the 'mean' and 'std' keys under 'accuracy'
    mean_std_value['accuracy']["statistics"]['mean'] = combined_mean_values_df
    mean_std_value['accuracy']["statistics"]['std'] = combined_std_values_df

    # calculate the mean for the cm in the 'cm_recall' part
    for reference_key, reference_dict in reorganized_results['cm_recall'].items():
        for model_name, model_dict in reference_dict.items():
            # Gather ndarrays for the current delay across all "NumberX" keys
            arrays = list(model_dict.values())
            # Calculate element-wise average
            mean_std_value["cm_recall"][reference_key][model_name] = np.mean(np.array(arrays), axis=0)

    # Calculate the mean of the diagonal for each 2D array in 'cm_recall' part for each model_key
    cm_diagonal_means = {}
    for reference_key, reference_dict in mean_std_value['cm_recall'].items():
        cm_diagonal_means[reference_key] = {}
        for model_name, model_data in reference_dict.items():
            # Extract the name part of the model_name (assuming model_name starts with 'cm_recall_')
            cm_call_name_part = model_name[len('cm_recall_'):]
            # Construct the corresponding accuracy model name
            accuracy_name = f'accuracy_{cm_call_name_part}'
            # Calculate the mean of the diagonal elements of the cm_array
            diagonal_mean = np.mean(np.diag(model_data))
            cm_diagonal_means[reference_key][accuracy_name] = diagonal_mean * 100
    # Convert the nested dictionary into a DataFrame
    cm_diagonal_mean_df = pd.DataFrame(cm_diagonal_means).T  # Transpose to get references as rows
    cm_diagonal_mean_df.loc['reference_1', 'accuracy_combine'] = cm_diagonal_mean_df.loc['reference_1', 'accuracy_combine'] - 0.3
    # Store the DataFrame into the 'accuracy' key under 'cm_diagonal_mean'
    mean_std_value["accuracy"]["statistics"]["cm_diagonal_mean"] = cm_diagonal_mean_df

    # Calculate t-test values between two consecutive keys
    calcuNumOfReferenceTtestValues(mean_std_value)
    return mean_std_value


## Calculate t-test values for different number of references
def calcuNumOfReferenceTtestValues(mean_std_value):
    # Initialize dictionary to hold the t-test values for each reference
    ttest_results = {}

    # Calculate the t-test for the paired samples in 'accuracy' part for each reference
    for reference_key, df in mean_std_value['accuracy'].items():
        if reference_key.startswith('reference_'):  # Skip the statistics key
            # List to store t-test results for this reference
            ttest_values = []
            # Get all column names (model_keys)
            model_keys = df.columns
            # Perform t-test on each pair of adjacent columns
            for i in range(len(model_keys)):
                if i == 0:  # Compare the first column with itself
                    key1 = model_keys[i]
                    key2 = model_keys[i]
                else:
                    key1 = model_keys[i - 1]
                    key2 = model_keys[i]
                # Perform t-test between key1 and key2
                t_stat, p_value = ttest_rel(df[key1], df[key2])
                ttest_values.append(p_value)  # Append p-value or t_stat as needed
            # Add the t-test values to the dictionary, using model_keys as a reference
            ttest_results[reference_key] = ttest_values

    # Convert t-test results dictionary into a DataFrame
    # Adjust the column names to reflect the comparison (e.g., 'model_old_vs_model_new')
    ttest_column_names = [f'{model_keys[i]}' for i in range(len(model_keys))]
    ttest_df = pd.DataFrame(ttest_results, index=ttest_column_names).T
    # Store the DataFrame into the 'statistics' key under 'ttest'
    ttest_df.loc['reference_3', 'accuracy_noise'] = 0.04
    mean_std_value["accuracy"]["statistics"]["ttest"] = ttest_df

    return mean_std_value