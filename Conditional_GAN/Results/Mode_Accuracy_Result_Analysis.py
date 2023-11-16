##
import numpy as np
import pandas as pd
import copy
from scipy.stats import ttest_rel


## extract accuracies of transitional modes in the cm_recall matrix from each model
def extractModeAccuracyFromCm(combined_results):
    # Define the positions to extract from the matrix
    positions = [(1, 1), (3, 3), (2, 2), (5, 5)]
    column_names = ["LWSA", "SALW", "LWSD", "SDLW"]

    # Initialize a dictionary to store the extracted data
    extracted_mode_accuracy = {"accuracy": {name: {} for name in column_names}}
    # Iterate over each model in the cm_recall dictionary
    for model_name, model_data in combined_results["cm_recall"].items():
        # Extract the required elements for each model
        for pos, name in zip(positions, column_names):
            extracted_values = [number_data[pos] * 100 for number_key, number_data in model_data["delay_0_ms"].items()]
            extracted_mode_accuracy["accuracy"][name][model_name] = dict(zip(model_data["delay_0_ms"].keys(), extracted_values))

    return extracted_mode_accuracy


## calculate statistical results for each transition mode
def CalcuModeAccuracyStataValues(reorganized_results):
    mean_std_value = copy.deepcopy(reorganized_results)

    # Compute the mean and std value of the accuracies
    mean_std_value["accuracy"]["statistics"] = {}
    # Initialize dictionaries to hold the mean and std values DataFrames
    mean_values_dict = {}
    std_values_dict = {}
    # Iterate over each reference in the 'accuracy' part of the dictionary
    for transition_mode, df in reorganized_results['accuracy'].items():
        # Calculate the mean and std of each column in the DataFrame
        mean_values = df.mean()
        std_values = df.std()
        # Convert the mean and std values Series to a single-row DataFrame
        mean_values_df = mean_values.to_frame().transpose()
        std_values_df = std_values.to_frame().transpose()
        # Assign the mean and std values DataFrames to the corresponding transition mode key
        mean_values_dict[transition_mode] = mean_values_df
        std_values_dict[transition_mode] = std_values_df
    # Combine all mean value DataFrames into a single DataFrame with reference_x as the index
    combined_mean_values_df = pd.concat(mean_values_dict)
    combined_std_values_df = pd.concat(std_values_dict)
    # Adjust the index to have only the reference_x as the index
    combined_mean_values_df.index = combined_mean_values_df.index.droplevel(1)
    combined_std_values_df.index = combined_std_values_df.index.droplevel(1)
    # Assign the combined mean and std values DataFrames to the 'mean' and 'std' keys under 'accuracy'
    mean_std_value['accuracy']["statistics"]['mean'] = combined_mean_values_df
    mean_std_value['accuracy']["statistics"]['std'] = combined_std_values_df

    # Calculate t-test values between two consecutive keys
    calcuNumOfReferenceTtestValues(mean_std_value)
    return mean_std_value

## Calculate t-test values for different number of references
def calcuNumOfReferenceTtestValues(mean_std_value):
    # Initialize dictionary to hold the t-test values for each reference
    ttest_results = {}

    # Calculate the t-test for the paired samples in 'accuracy' part for each reference
    for reference_key, df in mean_std_value['accuracy'].items():
        if reference_key != "statistics":  # Skip the statistics key
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

