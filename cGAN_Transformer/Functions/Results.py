
##
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

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

