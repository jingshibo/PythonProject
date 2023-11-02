##
from Conditional_GAN.Models import Model_Storage
import numpy as np
import pandas as pd
import copy
from scipy.stats import ttest_rel


## load results from all models of the subject
def getSubjectResults(subject, version, result_set):
    accuracy_basis, cm_recall_basis = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_basis',
    project='cGAN_Model')
    accuracy_old, cm_recall_old = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_old', project='cGAN_Model')

    accuracy_best, cm_recall_best = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_best', project='cGAN_Model')
    accuracy_worst, cm_recall_worst = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_worst', project='cGAN_Model')
    accuracy_new, cm_recall_new = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_new', project='cGAN_Model')
    accuracy_compare, cm_recall_compare = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_compare', project='cGAN_Model')
    accuracy_noise, cm_recall_noise = Model_Storage.loadClassifyResult(subject, version, result_set, 'classify_noise', project='cGAN_Model')

    accuracy = {'accuracy_best': accuracy_best, 'accuracy_new': accuracy_new, 'accuracy_compare': accuracy_compare,
        'accuracy_noise': accuracy_noise, 'accuracy_worst': accuracy_worst, 'accuracy_basis': accuracy_basis, 'accuracy_old': accuracy_old}
    cm_call = {'cm_recall_best': cm_recall_best, 'cm_recall_new': cm_recall_new, 'cm_recall_compare': cm_recall_compare,
        'cm_recall_noise': cm_recall_noise, 'cm_recall_worst': cm_recall_worst, 'cm_recall_basis': cm_recall_basis,
        'cm_recall_old': cm_recall_old}
    classify_results = {'accuracy': accuracy, 'cm_call': cm_call}
    return classify_results


## combine the results from all subjects into the dicts
def combineModelUpdateResults(original_data):
    combined_data = {"accuracy": {}, "cm_call": {}}
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
    mean_std_value["accuracy"]["mean"] = mean_df
    mean_std_value["accuracy"]["std"] = std_df

    # Calculate for cm_call dictionary
    for model_name, model_value in combined_results["cm_call"].items():
        mean_std_value["cm_call"][model_name] = {}
        for delay_key, delay_values in model_value.items():
            # Gather ndarrays for the current delay across all "NumberX" keys
            arrays = list(delay_values.values())
            # Calculate element-wise average
            mean_std_value["cm_call"][model_name][delay_key] = np.mean(np.array(arrays), axis=0)

    # Calculate t-test values between two consecutive keys
    computeTtestValuse(mean_std_value)

    return mean_std_value


def computeTtestValuse(mean_std_value):
    # Order of the keys as manually provided
    key_order = ['accuracy_best', 'accuracy_new', 'accuracy_compare', 'accuracy_noise', 'accuracy_worst']
    # Obtain the key order directly from the dictionary
    # key_order = list(mean_std_value["accuracy"].keys())
    # key_order = [key for key in key_order if key != "ttest"]  # Exclude the 'ttest' key if 'ttest' already exists in the 'accuracy' dict
    ttest_df = pd.DataFrame(index=mean_std_value["accuracy"][key_order[0]].columns)

    # Loop over consecutive keys
    for i in range(len(key_order)):
        if i == 0:  # key_order[0] = 'accuracy_best'
            key1 = key_order[i]
            key2 = key_order[i]
        else:
            key1 = key_order[i - 1]
            key2 = key_order[i]

        # check if the row index name starts with the word "Number" using Python's built-in string methods
        valid_rows = mean_std_value["accuracy"][key1].index[mean_std_value["accuracy"][key1].index.str.startswith("Number")]
        # Select rows only with index names that start with "Number"
        df1 = mean_std_value["accuracy"][key1].loc[valid_rows]
        df2 = mean_std_value["accuracy"][key2].loc[valid_rows]
        print(valid_rows)

        # Conduct paired t-test for each delay column and store p-values in ttest_df
        p_values = []
        for column in df1.columns:
            _, p_val = ttest_rel(df1[column], df2[column])
            p_values.append(p_val)
        ttest_df[key2] = p_values
    # Embed the ttest_df in the original transformed_data dictionary
    mean_std_value["accuracy"]["ttest"] = ttest_df


# exclude 'mean' and 'std', only Number...