'''
plot confusion matrix. normalize means calculate and plot the recall matrix.
'''

import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.patches import Rectangle


def plotConfusionMatrix(cm, classes, normalize=False, title='', cmap=plt.cm.Blues):
    cm_values = cm * 100
    plt.figure()
    plt.imshow(cm_values, interpolation='nearest', cmap=cmap)
    plt.title(title)
    font_size = 25
    color_bar = plt.colorbar()
    color_bar.ax.tick_params(labelsize=font_size)

    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45, fontsize=font_size)

    plt.xticks(tick_marks, classes, fontsize=font_size)
    plt.yticks(tick_marks, classes, fontsize=font_size)

    # Add separation lines by drawing rectangles around each cell
    # For inner grid lines
    for i in range(cm_values.shape[0]):
        for j in range(cm_values.shape[1]):
            plt.gca().add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=0.5))
    # For outer boundary
    plt.gca().add_patch(Rectangle((-0.5, -0.5), len(classes), len(classes), fill=False, edgecolor='black', linewidth=1))


    if normalize:
        cm_values = np.around(cm_values.astype('float') / cm_values.sum(axis=1)[:, np.newaxis], 1)  # calculate cm recall
        print("Confusion matrix, with normalization")
    else:
        cm_values = np.around(cm_values, 1)
        print('Confusion matrix, without normalization')

    thresh = cm_values.max() / 2
    for i, j in itertools.product(range(cm_values.shape[0]), range(cm_values.shape[1])):
        plt.text(j, i, cm_values[i, j], horizontalalignment="center", color="white" if cm_values[i, j] > thresh else "black", fontsize=font_size)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=font_size)
    plt.xlabel('Predicted label', fontsize=font_size)
    return cm_values
