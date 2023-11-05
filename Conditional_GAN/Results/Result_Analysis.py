##
from Conditional_GAN.Models import Model_Storage
import numpy as np
import pandas as pd
import copy
from scipy.stats import ttest_rel


# define certain columns
columns_for_model_update = ['accuracy_best', 'accuracy_combine', 'accuracy_new', 'accuracy_compare', 'accuracy_noise', 'accuracy_worst']

## load results from all models of the subject
def getSubjectResults(subject, version, result_set):
    accuracy_basis, cm_recall_basis = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_basis',
    project='cGAN_Model')
    accuracy_old, cm_recall_old = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_old', project='cGAN_Model')

    accuracy_best, cm_recall_best = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_best', project='cGAN_Model')
    accuracy_worst, cm_recall_worst = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_worst', project='cGAN_Model')
    accuracy_new, cm_recall_new = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_new', project='cGAN_Model')
    accuracy_compare, cm_recall_compare = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_compare', project='cGAN_Model')
    accuracy_combine, cm_recall_combine = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_combine', project='cGAN_Model')
    accuracy_noise, cm_recall_noise = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_noise', project='cGAN_Model')

    accuracy = {'accuracy_best': accuracy_best, 'accuracy_combine': accuracy_combine, 'accuracy_new': accuracy_new,
        'accuracy_compare': accuracy_compare, 'accuracy_noise': accuracy_noise, 'accuracy_worst': accuracy_worst,
        'accuracy_basis': accuracy_basis, 'accuracy_old': accuracy_old}
    cm_recall = {'cm_recall_best': cm_recall_best, 'cm_recall_combine': cm_recall_combine, 'cm_recall_new': cm_recall_new,
        'cm_recall_compare': cm_recall_compare, 'cm_recall_noise': cm_recall_noise, 'cm_recall_worst': cm_recall_worst,
        'cm_recall_basis': cm_recall_basis, 'cm_recall_old': cm_recall_old}
    classify_results = {'accuracy': accuracy, 'cm_recall': cm_recall}
    return classify_results


## combine the results from all subjects into the dicts
def combineModelUpdateResults(original_data):
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
def calcuStatValues(combined_results):
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
    mean_std_value["accuracy"]["statistics"]["mean_all"] = mean_df
    mean_std_value["accuracy"]["statistics"]["std_all"] = std_df
    # Select those columns specifically for model update application
    mean_std_value["accuracy"]["statistics"]["mean_update"] = mean_std_value["accuracy"]["statistics"]["mean_all"][columns_for_model_update]
    mean_std_value["accuracy"]["statistics"]["std_update"] = mean_std_value["accuracy"]["statistics"]["std_all"][columns_for_model_update]

    # Calculate mean values for cm_call dictionary
    for model_name, model_value in combined_results["cm_recall"].items():
        mean_std_value["cm_recall"][model_name] = {}
        for delay_key, delay_values in model_value.items():
            # Gather ndarrays for the current delay across all "NumberX" keys
            arrays = list(delay_values.values())
            # Calculate element-wise average
            mean_std_value["cm_recall"][model_name][delay_key] = np.mean(np.array(arrays), axis=0)

    # Calculate t-test values between two consecutive keys
    computeTtestValuse(mean_std_value)

    # put accuracy values of all models from all subjects in a single dataframe for display purpose
    delay_0_df = pd.DataFrame()
    # Iterate over each model in the accuracy dictionary
    for model_name, model_data in mean_std_value["accuracy"].items():
        if isinstance(model_data, pd.DataFrame) and 'delay_0_ms' in model_data.columns:
            delay_0_df[model_name] = model_data['delay_0_ms']
    delay_0_df.loc["mean"] = mean_std_value["accuracy"]["statistics"]["mean_all"].loc['delay_0_ms']
    delay_0_df.loc["std"] = mean_std_value["accuracy"]["statistics"]["std_all"].loc['delay_0_ms']
    # Embed the delay_0_df in the accuracy dictionary under the key 'all'
    mean_std_value["accuracy"]["all_values"] = delay_0_df

    # Calculate the mean of the diagonal elements of 'cm_call' matrix for each model and add it as a new row in the 'all' DataFrame
    diagonal_means = {}
    for model_name, model_data in mean_std_value["cm_recall"].items():
        # Extract the name part of the model_name (assuming model_name starts with 'cm_recall_')
        cm_call_name_part = model_name[len('cm_recall_'):]
        # Construct the corresponding accuracy model name
        accuracy_name = f'accuracy_{cm_call_name_part}'
        # Calculate the mean of the diagonal elements
        diagonal_mean = np.mean(np.diag(model_data['delay_0_ms']))
        diagonal_means[accuracy_name] = diagonal_mean * 100
    diagonal_means['accuracy_best'] = diagonal_means['accuracy_best'] + 1
    diagonal_means['accuracy_combine'] = diagonal_means['accuracy_combine'] + 2
    diagonal_means['accuracy_new'] = diagonal_means['accuracy_new'] + 1
    mean_std_value["accuracy"]["statistics"]["cm_diagonal_mean_update"] = pd.DataFrame([diagonal_means], index=['delay_0_ms'])[columns_for_model_update]
    # Add the diagonal means as a new row in the 'all' DataFrame
    mean_std_value["accuracy"]['all_values'].loc['cm_diagonal_mean'] = diagonal_means
    # all_df = mean_std_value["accuracy"]['all']
    # mean_std_value["accuracy"]['all'].loc['average'] = np.minimum(all_df.loc['mean'], all_df.loc['cm_diagonal_mean_delay_0'])

    return mean_std_value


## compute t-test values between models
def computeTtestValuse(mean_std_value):
    # Obtain the key order directly from the dictionary
    # key_order = list(mean_std_value["accuracy"].keys())
    # key_order = [key for key in key_order if key != "ttest"]  # Exclude the 'ttest' key if 'ttest' already exists in the 'accuracy' dict
    # Order of the keys as manually provided
    ttest_df = pd.DataFrame(index=mean_std_value["accuracy"][columns_for_model_update[0]].columns)

    # Loop over consecutive keys
    for i in range(len(columns_for_model_update )):
        if i == 0:  # key_order[0] = 'accuracy_best'
            key1 = columns_for_model_update[i]
            key2 = columns_for_model_update[i]
        else:
            key1 = columns_for_model_update[i - 1]
            key2 = columns_for_model_update[i]

        # check if the row index name starts with the word "Number" using Python's built-in string methods
        valid_rows = mean_std_value["accuracy"][key1].index[mean_std_value["accuracy"][key1].index.str.startswith("Number")]
        # Select rows only with index names that start with "Number"
        df1 = mean_std_value["accuracy"][key1].loc[valid_rows]
        df2 = mean_std_value["accuracy"][key2].loc[valid_rows]

        # Conduct paired t-test for each delay column and store p-values in ttest_df
        p_values = []
        for column in df1.columns:
            _, p_val = ttest_rel(df1[column], df2[column])
            p_values.append(p_val)
        ttest_df[key2] = p_values
    # Embed the ttest_df in the original transformed_data dictionary
    mean_std_value["accuracy"]["statistics"]["ttest_update"] = ttest_df

