##
from Conditional_GAN.Results import Result_Analysis, Plot_Statistics
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt


##
all_subjects = {}

##
subject = 'Number0'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number1'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number2'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
# subject = 'Number3'
# version = 0
# result_set = 0
# all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number4'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number5'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number6'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number7'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)

##
subject = 'Number8'
version = 0
result_set = 0
all_subjects[subject] = Result_Analysis.getSubjectResults(subject, version, result_set)


## combine the results
combined_results = Result_Analysis.combineSubjectResults(all_subjects)

## calcualte statistic values
mean_std_value = Result_Analysis.calcuSubjectStatValues(combined_results)


## plot accuracy
columns_for_plotting = ['accuracy_best', 'accuracy_tf', 'accuracy_combine', 'accuracy_new', 'accuracy_compare', 'accuracy_noise',
    'accuracy_copy', 'accuracy_worst']
legend = ['All New Data', 'All New TF', 'Add Synthetic + Old Data', 'Add Synthetic Data', 'Add Old Data', 'Add Noise', 'No Augmentation', 'No Update']
# legend = ['All New Data', 'Add Synthetic + Old Data', 'Add Synthetic Data', 'Add Old Data', 'Add Noise', 'No Augmentation', 'No Update']
title = 'Comparing Classification Accuracy of Different Methods for Model Updating '
Plot_Statistics.plotSubjectAdjacentTtest(mean_std_value, legend, columns_for_plotting, title=title, bonferroni_coeff=1)

##  plot confusion matrix
class_labels = ['LW-LW', 'LW-SA', 'LW-SD', 'SA-LW', 'SA-SA', 'SD-LW', 'SD-SD']
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_best']['delay_0_ms'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_tf']['delay_0_ms'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_combine']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_new']['delay_0_ms'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_compare']['delay_0_ms'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_noise']['delay_0_ms'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_worst']['delay_0_ms'], class_labels, normalize=False)

