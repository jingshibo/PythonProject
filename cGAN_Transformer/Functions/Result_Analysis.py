
##
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from cGAN_Transformer.Functions import Storage
import pandas as pd
import copy
from scipy.stats import ttest_rel


##
def getAccuracyCm(model_results):
    group_accuracies = []
    all_true = []
    all_pred = []
    for group in model_results:
        y_true = group['true_value']
        y_pred = group['predict_value']

        # Accuracy per group
        acc = accuracy_score(y_true, y_pred)
        group_accuracies.append(acc)

        # Collect for overall confusion matrix
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    # Convert to arrays
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Average accuracy across groups
    avg_accuracy = np.mean(group_accuracies)
    # Confusion matrix
    conf_mat = confusion_matrix(all_true, all_pred)
    cm_recall = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], 3)  # calculate cm recall

    return avg_accuracy, cm_recall


## load results from all models of the subject
def getSubjectResults(subject, version, result_set, gen_model, num_reference=1):
    # model update results
    accuracy_basis, cm_recall_basis = Storage.loadClassifyResult(subject, version, result_set, 'classify_basis', gen_model,
        project='cGAN_Model')
    accuracy_best, cm_recall_best = Storage.loadClassifyResult(subject, version, result_set, 'classify_best', gen_model,
        project='cGAN_Model')
    accuracy_tf, cm_recall_tf = Storage.loadClassifyResult(subject, version, result_set, 'classify_tf', gen_model,
        project='cGAN_Model', num_reference=10)
    accuracy_worst, cm_recall_worst = Storage.loadClassifyResult(subject, version, result_set, 'classify_worst', gen_model,
        project='cGAN_Model')

    accuracy_old, cm_recall_old = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_old', gen_model,
        project='cGAN_Model', num_reference=num_reference)
    accuracy_synthetic, cm_recall_synthetic = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_synthetic', gen_model,
        project='cGAN_Model', num_reference=num_reference)
    if num_reference != 0:
        accuracy_noise, cm_recall_noise = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_noisy', gen_model,
            project='cGAN_Model', num_reference=num_reference)
        accuracy_copy, cm_recall_copy = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_copy', gen_model,
            project='cGAN_Model', num_reference=num_reference)
    else:
        accuracy_noise, cm_recall_noise = (0, np.zeros_like(cm_recall_synthetic))
        accuracy_copy, cm_recall_copy = (0, np.zeros_like(cm_recall_synthetic))

    # old model results
    accuracy_old_real, cm_recall_old_real = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_real', gen_model,
        project='cGAN_Model')
    accuracy_old_synthetic, cm_recall_old_synthetic = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_synthetic', gen_model,
        project='cGAN_Model')
    accuracy_old_imbalance, cm_recall_old_imbalance = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_imbalance', gen_model,
        project='cGAN_Model', num_reference=5)
    accuracy_old_rebalanced, cm_recall_old_rebalanced = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_rebalanced', gen_model,
        project='cGAN_Model', num_reference=5)
    accuracy_old_noisy, cm_recall_old_noisy = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_noisy',
        gen_model, project='cGAN_Model', num_reference=5)

    accuracy = {'accuracy_old_real': accuracy_old_real, 'accuracy_old_synthetic': accuracy_old_synthetic,
        'accuracy_old_imbalance': accuracy_old_imbalance, 'accuracy_old_rebalanced': accuracy_old_rebalanced,
        'accuracy_old_noisy': accuracy_old_noisy, 'accuracy_best': accuracy_best, 'accuracy_tf': accuracy_tf,
        'accuracy_synthetic': accuracy_synthetic, 'accuracy_copy': accuracy_copy, 'accuracy_noise': accuracy_noise,
        'accuracy_old': accuracy_old, 'accuracy_worst': accuracy_worst, 'accuracy_basis': accuracy_basis}
    cm_recall = {'cm_recall_old_real': cm_recall_old_real, 'cm_recall_old_synthetic': cm_recall_old_synthetic,
        'cm_recall_old_imbalance': cm_recall_old_imbalance, 'cm_recall_old_rebalanced': cm_recall_old_rebalanced,
        'cm_recall_old_noisy': cm_recall_old_noisy, 'cm_recall_best': cm_recall_best, 'cm_recall_tf': cm_recall_tf,
        'cm_recall_synthetic': cm_recall_synthetic, 'cm_recall_copy': cm_recall_copy, 'cm_recall_noise': cm_recall_noise,
        'cm_recall_old': cm_recall_old, 'cm_recall_worst': cm_recall_worst, 'cm_recall_basis': cm_recall_basis}
    classify_results = {'accuracy': accuracy, 'cm_recall': cm_recall}

    return classify_results


##
def getOldResults(subject, version, gen_model, result_set):
    accuracy_old_real, cm_recall_old_real = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_real', gen_model,
        project='cGAN_Model')
    accuracy_old_rebalanced, cm_recall_old_rebalanced = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_rebalanced',
        gen_model, project='cGAN_Model', num_reference=5)
    accuracy_old_synthetic, cm_recall_old_synthetic = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_synthetic',
        gen_model, project='cGAN_Model')
    accuracy_old_imbalance, cm_recall_old_imbalance = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_imbalance',
        gen_model, project='cGAN_Model', num_reference=5)
    accuracy_old_noisy, cm_recall_old_noisy = Storage.loadClassifyResult(subject, version, result_set, 'classify_old_noisy',
        gen_model, project='cGAN_Model', num_reference=5)
    accuracy = {'accuracy_old_real': accuracy_old_real, 'accuracy_old_rebalanced': accuracy_old_rebalanced,
        'accuracy_old_synthetic': accuracy_old_synthetic, 'accuracy_old_imbalance': accuracy_old_imbalance,
        'accuracy_old_noisy': accuracy_old_noisy}
    cm_recall = {'cm_recall_old_real': cm_recall_old_real, 'cm_recall_old_rebalanced': cm_recall_old_rebalanced,
        'cm_recall_old_synthetic': cm_recall_old_synthetic, 'cm_recall_old_imbalance': cm_recall_old_imbalance,
        'cm_recall_old_noisy': cm_recall_old_noisy}

    classify_results = {'accuracy': accuracy, 'cm_recall': cm_recall}

    return classify_results

## load results from benchmark datasets of the subject (worst, best, tf_best, old_best)
def getBenchmarkResults(subject, version, gen_model, result_set):
    accuracy_basis, cm_recall_basis = Storage.loadClassifyResult(subject, version, result_set, 'classify_basis', gen_model,
        project='cGAN_Model')
    accuracy_best, cm_recall_best = Storage.loadClassifyResult(subject, version, result_set, 'classify_best', gen_model,
        project='cGAN_Model')
    accuracy_tf, cm_recall_tf = Storage.loadClassifyResult(subject, version, result_set, 'classify_tf', gen_model,
        project='cGAN_Model', num_reference=10)
    accuracy_worst, cm_recall_worst = Storage.loadClassifyResult(subject, version, result_set, 'classify_worst', gen_model,
        project='cGAN_Model')

    accuracy = {'accuracy_worst': accuracy_worst, 'accuracy_best': accuracy_best, 'accuracy_tf': accuracy_tf,
        'accuracy_basis': accuracy_basis}
    cm_recall = {'cm_recall_worst': cm_recall_worst, 'cm_recall_best': cm_recall_best, 'cm_recall_tf': cm_recall_tf,
        'cm_recall_basis': cm_recall_basis}
    classify_results = {'accuracy': accuracy, 'cm_recall': cm_recall}

    return classify_results


## load results of difference reference number for the subject
def getNumOfReferenceResults(subject, version, result_set, gen_model, num_references):
    classify_results = {'accuracy': {}, 'cm_recall': {}}
    for reference in num_references:
        accuracy_old, cm_recall_old = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_old', gen_model, project='cGAN_Model',
            num_reference=reference)
        accuracy_synthetic, cm_recall_synthetic = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_synthetic', gen_model,
            project='cGAN_Model', num_reference=reference)
        if reference != 0:
            accuracy_noise, cm_recall_noise = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_noisy', gen_model,
                project='cGAN_Model', num_reference=reference)
            accuracy_copy, cm_recall_copy = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_copy', gen_model,
                project='cGAN_Model', num_reference=reference)
        else:
            accuracy_noise, cm_recall_noise = (0, np.zeros_like(cm_recall_synthetic))
            accuracy_copy, cm_recall_copy = (0, np.zeros_like(cm_recall_synthetic))

        accuracy = {'accuracy_synthetic': accuracy_synthetic, 'accuracy_old': accuracy_old, 'accuracy_copy': accuracy_copy,
            'accuracy_noise': accuracy_noise}
        cm_recall = {'cm_recall_synthetic': cm_recall_synthetic, 'cm_recall_old': cm_recall_old, 'cm_recall_copy': cm_recall_copy,
            'cm_recall_noise': cm_recall_noise}

        classify_results['accuracy'][f'reference_{reference}'] = accuracy
        classify_results['cm_recall'][f'reference_{reference}'] = cm_recall

    return classify_results


## load only old results of different reference number for the subject
def getModeAccuracyResults(subject, version, result_set, gen_model, num_references):
    classify_results = {'accuracy': {}, 'cm_recall': {}}

    accuracy_tf, cm_recall_tf = Storage.loadClassifyResult(subject, version, result_set, 'classify_tf', gen_model,
        project='cGAN_Model', num_reference=10)
    classify_results['accuracy'][f'accuracy_tf'] = accuracy_tf
    classify_results['cm_recall'][f'cm_recall_tf'] = cm_recall_tf

    for reference in num_references:
        accuracy_synthetic, cm_recall_synthetic = Storage.loadClassifyResult(subject, version, result_set, 'classify_with_synthetic', gen_model,
            project='cGAN_Model', num_reference=reference)
        classify_results['accuracy'][f'accuracy_synthetic_{reference}'] = accuracy_synthetic
        classify_results['cm_recall'][f'cm_recall_synthetic_{reference}'] = cm_recall_synthetic

    accuracy_worst, cm_recall_worst = Storage.loadClassifyResult(subject, version, result_set, 'classify_worst', gen_model,
        project='cGAN_Model')
    classify_results['accuracy'][f'accuracy_worst'] = accuracy_worst
    classify_results['cm_recall'][f'cm_recall_worst'] = cm_recall_worst

    accuracy_best, cm_recall_best = Storage.loadClassifyResult(subject, version, result_set, 'classify_best', gen_model,
        project='cGAN_Model')
    classify_results['accuracy'][f'accuracy_best'] = accuracy_best
    classify_results['cm_recall'][f'cm_recall_best'] = cm_recall_best

    return classify_results


## combine the results from all subjects into the dicts
def combineSubjectResults(all_subjects):
    combined_data = {}

    for subject_key, subject_data in all_subjects.items():
        for metric_type in ['accuracy', 'cm_recall']:
            if metric_type in subject_data:
                if metric_type not in combined_data:
                    combined_data[metric_type] = {}
                for metric_key, value in subject_data[metric_type].items():
                    if metric_key not in combined_data[metric_type]:
                        combined_data[metric_type][metric_key] = {}
                    combined_data[metric_type][metric_key][subject_key] = value

    return combined_data


##
def calcuSubjectStatValues(combined_results):
    """
    Calculates statistical values (mean, std, etc.) for a results dictionary
    that does not have a 'delay' key level.

    Args:
        combined_results (dict): The nested dictionary containing 'accuracy' and 'cm_recall' data.

    Returns:
        dict: A new dictionary with added statistical results.
    """
    # Deepcopy to avoid modifying the original data structure
    stats_results = copy.deepcopy(combined_results)

    # --- 1. Calculate Mean & Std for Accuracy ---
    stats_results["accuracy"]["statistics"] = {}

    # Create a DataFrame from the accuracy data
    # This handles potentially different 'NumberX' keys between models
    all_accuracies_df = pd.DataFrame({model_name: pd.Series(model_data) for model_name, model_data in combined_results["accuracy"].items()})
    # Remove non-data rows like '__len__'
    all_accuracies_df = all_accuracies_df.drop('__len__', errors='ignore')

    # Calculate mean and std for each model (column)
    model_means = all_accuracies_df.mean().to_dict()
    model_stds = all_accuracies_df.std().to_dict()

    # Store the calculated statistics
    stats_results["accuracy"]["statistics"]["mean"] = pd.DataFrame([model_means], index=['mean']) * 100
    stats_results["accuracy"]["statistics"]["std"] = pd.DataFrame([model_stds], index=['std'])

    # --- 2. Calculate Mean for Confusion Matrices (cm_recall) ---
    for model_name, model_value in combined_results["cm_recall"].items():
        # Filter out non-array items like '__len__' and collect all ndarrays
        arrays = [v for k, v in model_value.items() if isinstance(v, np.ndarray)]

        if arrays:
            # Calculate element-wise average and replace the original dict
            stats_results["cm_recall"][model_name] = np.mean(np.array(arrays), axis=0)

    # --- 3. Create a single summary DataFrame for all accuracy values ---
    summary_df = all_accuracies_df.copy()
    summary_df.loc["mean"] = model_means
    summary_df.loc["std"] = model_stds
    stats_results["accuracy"]["all_values"] = summary_df

    # --- 4. T-test placeholder (as in original function) ---
    computeSubjectTtestValuse(stats_results)  # This would be called here if implemented

    # --- 5. Calculate mean of the diagonal of the mean 'cm_recall' matrices ---
    diagonal_means = {}
    for model_name, mean_cm_matrix in stats_results["cm_recall"].items():
        # Ensure it's a numpy array before processing
        if isinstance(mean_cm_matrix, np.ndarray):
            # Map cm_recall model name to accuracy model name
            # e.g., 'cm_recall_old_real' -> 'accuracy_old_real'
            accuracy_name = f"accuracy_{model_name[len('cm_recall_'):]}"

            # Calculate the mean of the diagonal elements
            diagonal_mean = np.mean(np.diag(mean_cm_matrix))
            diagonal_means[accuracy_name] = diagonal_mean * 100  # As percentage

    # --- 6. Add diagonal means to the results ---
    # Add as a new key in the statistics dict
    stats_results["accuracy"]["statistics"]["cm_diagonal_mean"] = pd.DataFrame([diagonal_means], index=['mean'])
    # Add as a new row in the summary DataFrame
    stats_results["accuracy"]['all_values'].loc['cm_diagonal_mean'] = pd.Series(diagonal_means)

    return stats_results


## compute t-test values between adjacent models
def computeSubjectTtestValuse(stats_results):
    """
    Computes a paired t-test between models, including comparing the first model
    to itself to ensure the output has a column for every model.

    Args:
        stats_results (dict): The dictionary produced by `calculate_statistics`.
                              It is modified in-place.
    """
    all_values_df = stats_results['accuracy']['all_values']

    # Filter to get only the raw subject data (rows starting with "Number")
    subject_data_df = all_values_df[all_values_df.index.str.startswith('Number')]

    model_keys = subject_data_df.columns.tolist()

    ttest_results = {}

    # --- This loop now replicates the original logic exactly ---
    # It iterates through ALL models, from the first to the last.
    for i in range(len(model_keys)):
        # The key change is here:
        if i == 0:
            # For the first model, compare it against itself.
            model1_key = model_keys[i]
            model2_key = model_keys[i]
        else:
            # For subsequent models, compare against the previous one.
            model1_key = model_keys[i - 1]
            model2_key = model_keys[i]

        # Get the accuracy scores for the two models being compared
        series1 = subject_data_df[model1_key]
        series2 = subject_data_df[model2_key]

        # Perform the paired t-test.
        # Note: ttest_rel(x, x) will result in a p-value of 0.0 if there's variance,
        # or NaN if all values are identical. This is the expected behavior.
        t_stat, p_val = ttest_rel(series1, series2)

        # The key for the result is always the second model in the comparison
        ttest_results[model2_key] = p_val

    # Convert the results dictionary to a DataFrame
    ttest_df = pd.DataFrame([ttest_results], index=['p_value'])

    # Embed the t-test DataFrame back into the main results dictionary
    stats_results["accuracy"]["statistics"]["ttest"] = ttest_df

    return stats_results


##
def combineNumOfReferenceResults(all_subjects_data):
    """
    Restructures a dictionary from being subject-centric to metric-centric.

    Input format:  data[subject][metric][reference][model] -> value
    Output format: data[metric][reference][model][subject] -> value

    Args:
        all_subjects_data (dict): The original dictionary, with subject keys
                                  at the top level.

    Returns:
        dict: The restructured dictionary.
    """
    # Initialize the primary keys ('accuracy', 'cm_recall') for the new dict
    output_keys = list(next(iter(all_subjects_data.values())).keys())
    combined_data = {key: {} for key in output_keys}

    # --- Loop through the original data structure ---

    # Level 1: Iterate over subjects ('Number0', 'Number1', etc.)
    for subject_key, subject_data in all_subjects_data.items():

        # Level 2: Iterate over metrics ('accuracy', 'cm_recall')
        for metric_key, metric_data in subject_data.items():

            # Level 3: Iterate over reference conditions ('reference_0', 'reference_1', etc.)
            for reference_key, reference_data in metric_data.items():

                # Level 4: Iterate over models ('accuracy_synthetic', 'accuracy_old', etc.)
                # In this structure, `value` is the final float or ndarray.
                for model_key, value in reference_data.items():
                    # --- Build the new, restructured dictionary ---

                    # Ensure the nested dictionaries exist.
                    # setdefault is a concise way to do this.
                    # It gets the key's value, or sets it to {} if it doesn't exist.
                    ref_dict = combined_data[metric_key].setdefault(reference_key, {})
                    model_dict = ref_dict.setdefault(model_key, {})

                    # Assign the value, with the subject key now at the innermost level.
                    model_dict[subject_key] = value

    return combined_data


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
            extracted_values = [number_data[pos] * 100 for number_key, number_data in model_data.items()]
            extracted_mode_accuracy["accuracy"][name][model_name] = dict(zip(model_data.keys(), extracted_values))

    return extracted_mode_accuracy