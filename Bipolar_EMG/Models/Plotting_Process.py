##
import numpy as np


##  average accuracies across subjects
def calculate_metrics(all_subjects):
    results = {}
    # Assuming all subjects have the same model types, use the first subject to get the model types
    model_types = all_subjects[next(iter(all_subjects))].keys()

    for model_type in model_types:
        accuracies = []
        cm_recalls = []
        cm_nums = []

        for subject in all_subjects.values():
            accuracies.append(subject[model_type]['accuracy'])
            cm_recalls.append(subject[model_type]['cm_recall'])
            cm_nums.append(subject[model_type]['cm_num'])

        # Calculate mean and std for accuracies
        accuracy_mean = np.mean(accuracies)
        accuracy_std = np.std(accuracies)

        # Calculate mean for cm_recall and cm_num
        cm_recall_mean = np.mean(np.stack(cm_recalls), axis=0)
        cm_num_mean = np.mean(np.stack(cm_nums), axis=0)

        results[model_type] = {
            'accuracy_mean': accuracy_mean,
            'accuracy_std': accuracy_std,
            'cm_recall_mean': cm_recall_mean,
            'cm_num_mean': cm_num_mean
        }

    return results


## mean value of the six single bipolar EMG
def calculate_mean_bipolar(results, bipolar_types):
    total_accuracy_mean = 0
    total_accuracy_std = 0
    total_cm_recall_mean = None
    total_cm_num_mean = None

    for bipolar in bipolar_types:
        total_accuracy_mean += results[bipolar]['accuracy_mean']
        total_accuracy_std += results[bipolar]['accuracy_std']

        if total_cm_recall_mean is None:
            total_cm_recall_mean = results[bipolar]['cm_recall_mean']
            total_cm_num_mean = results[bipolar]['cm_num_mean']
        else:
            total_cm_recall_mean += results[bipolar]['cm_recall_mean']
            total_cm_num_mean += results[bipolar]['cm_num_mean']

    # Calculate the mean for each metric
    average_accuracy_mean = total_accuracy_mean / len(bipolar_types)
    average_accuracy_std = total_accuracy_std / len(bipolar_types)
    average_cm_recall_mean = total_cm_recall_mean / len(bipolar_types)
    average_cm_num_mean = total_cm_num_mean / len(bipolar_types)

    average_bipolar = {'accuracy_mean': average_accuracy_mean, 'accuracy_std': average_accuracy_std,
        'cm_recall_mean': average_cm_recall_mean, 'cm_num_mean': average_cm_num_mean}
    results['AB_0'] = average_bipolar