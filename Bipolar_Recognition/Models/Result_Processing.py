import os
import json
import numpy as np


## save classification accuracy and cm recall values
def saveResult(subject, average_accuracies, average_cm_numbers, average_cm_recalls, model_type, result_set, project='HDsEMG_Recognition'):
    data_dir = f'D:\Data\{project}\subject_{subject}\\results'
    result_file = f'subject_{subject}_model_{model_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    # Combine the two dictionaries into one
    combined_results = {'accuracy': average_accuracies, 'cm_num': [arr.tolist() for arr in average_cm_numbers],
        'cm_recall': [arr.tolist() for arr in average_cm_recalls]}

    # Save to JSON file
    with open(result_path, 'w') as f:
        json.dump(combined_results, f, indent=8)


## read classification accuracy and cm recall values
def loadResult(subject, model_type, result_set, project='HDsEMG_Recognition'):
    data_dir = f'D:\Data\{project}\subject_{subject}\\results'
    result_file = f'subject_{subject}_model_{model_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    with open(result_path, 'r') as f:
        loaded_data = json.load(f)

    combined_results = {'accuracy': loaded_data['accuracy'], 'cm_recall': [np.array(lst) for lst in loaded_data['cm_recall']],
        'cm_num': [np.array(lst) for lst in loaded_data['cm_num']]}
    return combined_results
