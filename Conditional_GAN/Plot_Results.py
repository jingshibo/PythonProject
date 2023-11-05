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



## reorganize the results
combined_results = Result_Analysis.combineModelUpdateResults(all_subjects)

## calcualte statistic values
mean_std_value = Result_Analysis.calcuStatValues(combined_results)


## plot accuracy
dataset = 'model_type'
legend = ['All New Data', 'Add Synthetic + Old Data', 'Add Synthetic Data', 'Add Old Data', 'Add Noise', 'No Augmentation', 'No Update']
legend = ['All New Data', 'Add Synthetic + Old Data', 'Add Synthetic Data', 'Add Old Data', 'Add Noise', 'No Update']
title = 'Comparing Classification Accuracy of Different Model Updating Methods'
Plot_Statistics.plotAdjacentTtest(mean_std_value, legend, title=title, bonferroni_coeff=1)


##  plot confusion matrix
# class_labels = ['LW-LW', 'LW-SA', 'LW-SD', 'SA-LW', 'SA-SA', 'SD-LW', 'SD-SD']
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_best']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_combine']['delay_0_ms'], class_labels, normalize=False)
# # Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_new']['delay_0_ms'], class_labels, normalize=False)
# # Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_compare']['delay_0_ms'], class_labels, normalize=False)
# # Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_noise']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_worst']['delay_0_ms'], class_labels, normalize=False)
