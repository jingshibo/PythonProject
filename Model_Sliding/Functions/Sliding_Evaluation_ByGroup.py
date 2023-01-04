import numpy as np
from sklearn.metrics import confusion_matrix
from Models.Utility_Functions import Confusion_Matrix
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

## calculate accuracy and cm values for each group
def getAccuracyPerGroup(sliding_prediction, reorganized_truevalues):
    accuracy = []
    cm = []
    for group_number, group_value in enumerate(sliding_prediction):
        transition_cm = {}
        transition_accuracy = {}
        for transition_type, transition_result in group_value.items():
            true_y = reorganized_truevalues[group_number][transition_type]
            predict_y = transition_result[0, :]
            numCorrect = np.count_nonzero(true_y == predict_y)
            transition_accuracy[transition_type] = numCorrect / len(true_y) * 100
            transition_cm[transition_type] = confusion_matrix(y_true=true_y, y_pred=predict_y)
        accuracy.append(transition_accuracy)
        cm.append(transition_cm)
    return accuracy, cm


## calculate average accuracy
def averageAccuracy(accuracy, cm):
    transition_groups = list(accuracy[0].keys())  # list all transition types
    # average accuracy for each transition type
    average_accuracy = {transition: 0 for transition in transition_groups}  # initialize average accuracy list
    for group_values in accuracy:
        for transition_type, transition_accuracy in group_values.items():
            average_accuracy[transition_type] = average_accuracy[transition_type] + transition_accuracy
    for transition_type, transition_accuracy in average_accuracy.items():
        average_accuracy[transition_type] = transition_accuracy / len(accuracy)

    # overall accuracy for all transition types
    overall_accuracy = (average_accuracy['transition_LW'] * 1.5 + average_accuracy['transition_SA'] + average_accuracy['transition_SD'] +
                        average_accuracy['transition_SS'] * 1.5) / 5

    # overall cm among groups
    sum_cm = {transition: 0 for transition in transition_groups}   # initialize overall cm list
    for group_values in cm:
        for transition_type, transition_cm in group_values.items():
            sum_cm[transition_type] = sum_cm[transition_type] + transition_cm

    return average_accuracy, overall_accuracy, sum_cm


## plot confusion matrix
def confusionMatrix(sum_cm, recall=False):
    # create a diagonal matrix including all categories from multiple arrays.
    list_cm = [cm for label, cm in sum_cm.items()]
    overall_cm = block_diag(*list_cm)

    # the label order in the classes list should correspond to the one hot labels, which is a alphabetical order
    class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'SALW', 'SASA', 'SASS', 'SDLW', 'SDSD', 'SDSS', 'SSLW', 'SSSA', 'SSSD', 'SSSS']
    # class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'SALW', 'SASA', 'SASS', 'SDLW', 'SDSD', 'SDSS', 'SSLW', 'SSSA', 'SSSD']
    plt.figure()
    cm_recall = Confusion_Matrix.plotConfusionMatrix(overall_cm, class_labels, normalize=recall)
    return cm_recall