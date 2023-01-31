'''
post processing using the majority vote method and calculate the accuracy and confusion matrix.
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from Models.Utility_Functions import Confusion_Matrix


## majority vote
def majorityVoteResults(classify_results, window_per_repetition):
    '''
    The majority vote results for each transition repetition
    '''
    bin_results = []
    for result in classify_results:  # reunite the samples from the same transition
        true_y = []
        predict_y = []
        for key, value in result.items():
            if key == 'true_value':
                for i in range(0, len(value), window_per_repetition):
                    true_y.append(value[i: i+window_per_repetition])
            elif key == 'predict_value':
                for i in range(0, len(value), window_per_repetition):
                    predict_y.append(value[i: i+window_per_repetition])
        bin_results.append({"true_value": true_y, "predict_value": predict_y})

    majority_results = []
    for result in bin_results:  # use majority vote to get a consensus result
        true_y = []
        predict_y = []
        for key, value in result.items():
            if key == 'true_value':
                true_y = [np.bincount(i).argmax() for i in value]
            elif key == 'predict_value':
                predict_y = [np.bincount(i).argmax() for i in value]
        majority_results.append({"true_value": np.array(true_y), "predict_value": np.array(predict_y)})
    return majority_results


## accuracy
def averageAccuracy(majority_results):
    '''
    The accuracy for each cross validation group and average value across groups
    '''
    cm = []
    accuracy = []
    for result in majority_results:
        true_y = result['true_value']
        predict_y = result['predict_value']
        num_Correct = np.count_nonzero(true_y == predict_y)
        accuracy.append(num_Correct / len(true_y) * 100)
        cm.append(confusion_matrix(y_true=true_y, y_pred=predict_y))
    mean_accuracy = sum(accuracy) / len(accuracy)
    sum_cm = np.sum(np.array(cm), axis=0)
    return mean_accuracy, sum_cm


## plot confusion matrix
def confusionMatrix(sum_cm, recall=False):
    # the label order in the classes list should correspond to the one hot labels, which is a alphabetical order
    # class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'SALW', 'SASA', 'SASS', 'SDLW', 'SDSD', 'SDSS', 'SSLW', 'SSSA', 'SSSD', 'SSSS']
    class_labels = ['LW-LW', 'LW-SA', 'LW-SD', 'LW-SS', 'SA-LW', 'SA-SA', 'SA-SS', 'SD-LW', 'SD-SD', 'SD-SS', 'SS-LW', 'SS-SA', 'SS-SD']
    plt.figure()
    cm_recall = Confusion_Matrix.plotConfusionMatrix(sum_cm, class_labels, normalize=recall)
    return cm_recall
