## import
import copy
import pandas as pd
from scipy.stats import ttest_rel


## calculate statistical results for each condition
def CalcuStatValues(combined_results):
    mean_std_value = copy.deepcopy(combined_results)
    for dataset, datavalue in combined_results.items():
        if dataset != 'model_type':  # always compare the results to the hdemg convrnn model
            for condition, results in datavalue.items():
                for item, value in results.items():
                    if item == 'accuracy':
                        mean_std_value[dataset][condition][item] = pd.DataFrame(value)
                        mean_std_value[dataset][condition]['mean'] = pd.DataFrame(value).mean()
                        mean_std_value[dataset][condition]['std'] = pd.DataFrame(value).std()
                        mean_std_value[dataset][condition]['ttest'] = {}
                        for col in mean_std_value[dataset][condition][item].columns:
                            t_stat, p_value = ttest_rel(mean_std_value[dataset][condition][item][col],
                                combined_results['model_type']['Raw_ConvRnn']['accuracy'][col])
                            mean_std_value[dataset][condition]['ttest'][col] = p_value
                        mean_std_value[dataset][condition]['ttest'] = pd.Series(mean_std_value[dataset][condition]['ttest'])
        elif dataset == 'model_type':  # always compare the results to the adjacent model
            # Define the order of the keys
            models = list(datavalue.keys())
            for i in range(0, len(models)):
                mean_std_value[dataset][models[i]]['accuracy'] = pd.DataFrame(datavalue[models[i]]['accuracy'])
                mean_std_value[dataset][models[i]]['mean'] = pd.DataFrame(datavalue[models[i]]['accuracy']).mean()
                mean_std_value[dataset][models[i]]['std'] = pd.DataFrame(datavalue[models[i]]['accuracy']).std()
                mean_std_value[dataset][models[i]]['ttest'] = {}
                for col in mean_std_value[dataset][models[i]]['accuracy'].columns:
                    if i == 0:  # models[0] == 'Raw_ConvRnn'
                        t_stat, p_value = ttest_rel(datavalue[models[i]]['accuracy'][col], datavalue[models[i]]['accuracy'][col])
                    else:
                        t_stat, p_value = ttest_rel(datavalue[models[i]]['accuracy'][col], datavalue[models[i-1]]['accuracy'][col])
                    mean_std_value[dataset][models[i]]['ttest'][col] = p_value
                mean_std_value[dataset][models[i]]['ttest'] = pd.Series(mean_std_value[dataset][models[i]]['ttest'])

    return mean_std_value


## reorganize mean, std and ttest values
def reorganizeStatResults(mean_std_value):
    # Define a list of keys to extract from each dictionary
    keys = ['mean', 'std', 'ttest']

    # Create a new dictionary to store the reorganized results
    reorganized_results = copy.deepcopy(mean_std_value)
    # Iterate through each dataset and extract the specified keys
    for dataset, datavalue in mean_std_value.items():
        # Create new keys in the dict for storage
        for key in keys:
            reorganized_results[dataset][key] = {}

        # put hdemg results into all other dataset for comparison, excluding the 'model_type' dataset
        if dataset != 'model_type':
            for key in keys:
                reorganized_results[dataset][key]['hdemg'] = mean_std_value['model_type']['Raw_ConvRnn'][key]
        # extract the statistical results from each condition to the dataset dict it belongs to
        for condition, results in datavalue.items():
            for key in keys:
                reorganized_results[dataset][key][condition] = results[key]
        # put bipoalr results into reduce dataset for comparison,excluding the 'model_type' dataset
        if dataset in ['reduce_density_dataset', 'reduce_muscle_dataset'] and dataset != 'model_type':
            for key in keys:
                reorganized_results[dataset][key]['bipolar'] = mean_std_value['reduce_area_dataset']['channel_area_2'][key]

        # convert each dictionary to a DataFrame
        for key in keys:
            reorganized_results[dataset][key] = pd.DataFrame(reorganized_results[dataset][key])

    return reorganized_results



