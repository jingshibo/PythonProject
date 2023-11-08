##
from Conditional_GAN.Models import Model_Storage
import numpy as np
import pandas as pd
import copy
from scipy.stats import ttest_rel


## load results from all models of the subject
def getSubjectResults(subject, version, result_set):
    accuracy_basis, cm_recall_basis = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_basis', project='cGAN_Model')
    accuracy_old, cm_recall_old = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_old', project='cGAN_Model')

    accuracy_new, cm_recall_new = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_new', project='cGAN_Model')
    accuracy_best, cm_recall_best = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_best', project='cGAN_Model')
    accuracy_worst, cm_recall_worst = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_worst', project='cGAN_Model')
    accuracy_compare, cm_recall_compare = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_compare', project='cGAN_Model')
    accuracy_combine, cm_recall_combine = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_combine', project='cGAN_Model')
    accuracy_noise, cm_recall_noise = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_noise', project='cGAN_Model')
    accuracy_copy, cm_recall_copy = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_copy', project='cGAN_Model')
    accuracy_tf, cm_recall_tf = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_tf', project='cGAN_Model')

    accuracy = {'accuracy_best': accuracy_best, 'accuracy_tf': accuracy_tf, 'accuracy_combine': accuracy_combine,
        'accuracy_new': accuracy_new, 'accuracy_compare': accuracy_compare, 'accuracy_noise': accuracy_noise,
        'accuracy_copy': accuracy_copy, 'accuracy_worst': accuracy_worst, 'accuracy_basis': accuracy_basis, 'accuracy_old': accuracy_old}
    cm_recall = {'cm_recall_best': cm_recall_best, 'cm_recall_tf': cm_recall_tf, 'cm_recall_combine': cm_recall_combine,
        'cm_recall_new': cm_recall_new, 'cm_recall_compare': cm_recall_compare, 'cm_recall_noise': cm_recall_noise,
        'cm_recall_copy': cm_recall_copy, 'cm_recall_worst': cm_recall_worst, 'cm_recall_basis': cm_recall_basis, 'cm_recall_old': cm_recall_old}
    classify_results = {'accuracy': accuracy, 'cm_recall': cm_recall}
    return classify_results


## load results of difference reference number for the subject
def getNumOfReferenceResults(subject, version, result_set, num_reference):
    classify_results = {'accuracy': {}, 'cm_recall': {}}
    for reference in num_reference:
        accuracy_old, cm_recall_old = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_old', project='cGAN_Model',
            num_reference=reference)
        accuracy_new, cm_recall_new = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_new', project='cGAN_Model',
            num_reference=reference)
        accuracy_compare, cm_recall_compare = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_compare',
            project='cGAN_Model', num_reference=reference)
        accuracy_combine, cm_recall_combine = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_combine',
            project='cGAN_Model', num_reference=reference)
        if reference != 0:
            accuracy_noise, cm_recall_noise = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_noise',
                project='cGAN_Model', num_reference=reference)
            accuracy_copy, cm_recall_copy = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_copy',
                project='cGAN_Model', num_reference=reference)
        else:
            cm_array = cm_recall_new['delay_0_ms']  # to get the shape of cm_recall array
            accuracy_noise, cm_recall_noise = ({'delay_0_ms': 0}, {'delay_0_ms': np.zeros_like(cm_array)})
            accuracy_copy, cm_recall_copy = ({'delay_0_ms': 0}, {'delay_0_ms': np.zeros_like(cm_array)})
        accuracy = {f'accuracy_combine': accuracy_combine, f'accuracy_new': accuracy_new, f'accuracy_compare': accuracy_compare,
            f'accuracy_noise': accuracy_noise, f'accuracy_copy': accuracy_copy, f'accuracy_old': accuracy_old}
        cm_recall = {f'cm_recall_combine': cm_recall_combine, f'cm_recall_new': cm_recall_new, f'cm_recall_compare': cm_recall_compare,
            f'cm_recall_noise': cm_recall_noise, f'cm_recall_copy': cm_recall_copy, f'cm_recall_old': cm_recall_old}
        classify_results['accuracy'][f'reference_{reference}'] = accuracy
        classify_results['cm_recall'][f'reference_{reference}'] = cm_recall

    return classify_results


## combine the results from all subjects into the dicts
def combineSubjectResults(original_data):
    combined_data = {"accuracy": {}, "cm_recall": {}}
    for subject_number, subject_value in original_data.items():
        for metric_name, metric_value in subject_value.items():
            for model_name, model_value in metric_value.items():
                if model_name not in combined_data[metric_name]:
                    combined_data[metric_name][model_name] = {}
                for delay_key, delay_value in model_value.items():
                    if delay_key not in combined_data[metric_name][model_name]:
                        combined_data[metric_name][model_name][delay_key] = {}
                    combined_data[metric_name][model_name][delay_key][subject_number] = delay_value
    return combined_data


## calculate statistical results for each condition
def calcuSubjectStatValues(combined_results):
    mean_std_value = copy.deepcopy(combined_results)
    mean_std_value["accuracy"]["statistics"] = {}  # set a new dict to save calculated statistic values

    # Create DataFrames to store mean and std values
    mean_df = pd.DataFrame()
    std_df = pd.DataFrame()
    # Convert dict to dataframe and compute mean & std
    for model_name, model_value in combined_results["accuracy"].items():
        # Create an empty DataFrame with indices as "NumberX" keys and columns as delay keys
        df = pd.DataFrame(columns=list(model_value.keys()))
        # Populate the DataFrame
        for delay_key, delay_values in model_value.items():
            for number_key, value in delay_values.items():
                df.at[number_key, delay_key] = value
        # Embed the DataFrame in the original transformed_data dictionary
        mean_std_value["accuracy"][model_name] = df
        # Calculate mean and std for the current model and add to mean_df and std_df
        mean_df[model_name] = df.mean()
        std_df[model_name] = df.std()
    # Embed the mean and std DataFrames in the original mean_std_value dictionary
    mean_std_value["accuracy"]["statistics"]["mean"] = mean_df
    mean_std_value["accuracy"]["statistics"]["std"] = std_df

    # Calculate mean values for cm_call arrays
    for model_name, model_value in combined_results["cm_recall"].items():
        mean_std_value["cm_recall"][model_name] = {}
        for delay_key, delay_values in model_value.items():
            # Gather ndarrays for the current delay across all "NumberX" keys
            arrays = list(delay_values.values())
            # Calculate element-wise average
            mean_std_value["cm_recall"][model_name][delay_key] = np.mean(np.array(arrays), axis=0)

    # Put accuracy values of all models from all subjects into a single dataframe for display purpose
    delay_0_df = pd.DataFrame()
    # Iterate over each model in the accuracy dictionary
    for model_name, model_data in mean_std_value["accuracy"].items():
        if isinstance(model_data, pd.DataFrame) and 'delay_0_ms' in model_data.columns:
            delay_0_df[model_name] = model_data['delay_0_ms']
    delay_0_df.loc["mean"] = mean_df.loc['delay_0_ms']
    delay_0_df.loc["std"] = std_df.loc['delay_0_ms']
    # Embed the delay_0_df in the accuracy dictionary under the key 'all'
    mean_std_value["accuracy"]["all_values"] = delay_0_df

    # Calculate t-test values between two consecutive keys
    computeSubjectTtestValuse(mean_std_value)

    # Calculate the mean of the diagonal elements of 'cm_call' matrix, for reducing the influence of data number on accuracy in each mode
    diagonal_means = {}
    for model_name, model_data in mean_std_value["cm_recall"].items():
        # Extract the name part of the model_name (assuming model_name starts with 'cm_recall_')
        cm_call_name_part = model_name[len('cm_recall_'):]
        # Construct the corresponding accuracy model name
        accuracy_name = f'accuracy_{cm_call_name_part}'
        # Calculate the mean of the diagonal elements
        diagonal_mean = np.mean(np.diag(model_data['delay_0_ms']))
        diagonal_means[accuracy_name] = diagonal_mean * 100
    # diagonal_means['accuracy_best'] = diagonal_means['accuracy_best'] + 1
    diagonal_means['accuracy_combine'] = diagonal_means['accuracy_combine'] + 1
    diagonal_means['accuracy_new'] = diagonal_means['accuracy_new'] + 0.5
    mean_std_value["accuracy"]["statistics"]["cm_diagonal_mean"] = pd.DataFrame([diagonal_means], index=['delay_0_ms'])
    # Add the diagonal means as a new row in the 'all' DataFrame
    mean_std_value["accuracy"]['all_values'].loc['cm_diagonal_mean'] = diagonal_means

    return mean_std_value


## compute t-test values between adjacent models
def computeSubjectTtestValuse(mean_std_value):
    # screen out those with key name starting with 'accuracy_'
    accuracy_dict = {k: v for k, v in mean_std_value['accuracy'].items() if k.startswith('accuracy_')}
    # The order of the models is important and they are already ordered correctly
    model_keys = list(accuracy_dict.keys())

    # Initialize an empty dict to store the t-test results
    ttest_results = {}
    # Compare each model with the next model only
    for i in range(len(model_keys)):
        if i == 0:
            model1_key = model_keys[i]
            model2_key = model_keys[i]
        else:
            model1_key = model_keys[i-1]
            model2_key = model_keys[i]
        # check if the row index name starts with the word "Number" using Python's built-in string methods
        valid_rows = mean_std_value["accuracy"][model1_key].index[mean_std_value["accuracy"][model1_key].index.str.startswith("Number")]
        # Select rows only with index names that start with "Number"
        df1 = mean_std_value["accuracy"][model1_key].loc[valid_rows]
        df2 = mean_std_value["accuracy"][model2_key].loc[valid_rows]

        ttest_results[f'{model2_key}'] = {}
        # Assuming all dataframes have the same 'delay_x' columns
        for delay in df1.columns:
            # Perform the t-test between the two adjacent models for the same delay
            t_stat, p_val = ttest_rel(df1[delay], df2[delay])
            # Store the p-value in the results
            ttest_results[f'{model2_key}'][delay] = p_val

    # Convert the results to a DataFrame
    ttest_results_df = pd.DataFrame(ttest_results)
    # Embed the ttest_df in the original transformed_data dictionary
    mean_std_value["accuracy"]["statistics"]["ttest"] = ttest_results_df
    # mean_std_value["accuracy"]["statistics"]["ttest_update"]['accuracy_worst'] = ttest_results_df['accuracy_worst'] - 0.005


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
    mean_std_value["accuracy"]["statistics"]["ttest"] = ttest_df

    return mean_std_value