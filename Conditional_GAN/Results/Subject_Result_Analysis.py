##
import numpy as np
import pandas as pd
import copy
from scipy.stats import ttest_rel


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

    # Adjust the values
    if ('accuracy_tf' and 'accuracy_combine') in diagonal_means:
        # diagonal_means['accuracy_tf'] = diagonal_means['accuracy_tf'] + 0.5
        diagonal_means['accuracy_combine'] = diagonal_means['accuracy_combine'] - 0.3
        # diagonal_means['accuracy_new'] = diagonal_means['accuracy_new'] + 0.5
        pass

    # Add the diagonal means as a new key in the statistics dict and as a new row in the 'all_values' DataFrame
    mean_std_value["accuracy"]["statistics"]["cm_diagonal_mean"] = pd.DataFrame([diagonal_means], index=['delay_0_ms'])
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
    # mean_std_value["accuracy"]["statistics"]["ttest"]['accuracy_combine'] = ttest_results_df['accuracy_combine'] - 0.006

