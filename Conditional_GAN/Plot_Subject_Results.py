'''
    plot the results for all subjects, using a certain number of references as an representative.
'''


##
from Conditional_GAN.Results import Load_Results, Subject_Result_Analysis, Plot_Statistics
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt


##
all_subjects = {}
version = 0
basis_result_set = 0  # corresponds to the filter size (ten times): e.g., filter size = 10 relates to the results_set = 1
filter_result_set = 2
num_reference = 1

##
subject = 'Number0'
all_subjects[subject] = Load_Results.getSubjectResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number1'
all_subjects[subject] = Load_Results.getSubjectResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number2'
all_subjects[subject] = Load_Results.getSubjectResults(subject, version, basis_result_set, filter_result_set, num_reference)

# ##
# subject = 'Number3'
# all_subjects[subject] = Load_Results.getSubjectResults(subject, version, basis_result_set, reference_result_set, num_reference)

##
subject = 'Number4'
all_subjects[subject] = Load_Results.getSubjectResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number5'
all_subjects[subject] = Load_Results.getSubjectResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number6'
all_subjects[subject] = Load_Results.getSubjectResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number7'
all_subjects[subject] = Load_Results.getSubjectResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number8'
all_subjects[subject] = Load_Results.getSubjectResults(subject, version, basis_result_set, filter_result_set, num_reference)


## combine the results
combined_results = Subject_Result_Analysis.combineSubjectResults(all_subjects)

## calcualte statistic values
mean_std_value = Subject_Result_Analysis.calcuSubjectStatValues(combined_results)


## plot accuracy
# columns_for_plotting = ['accuracy_best', 'accuracy_tf', 'accuracy_combine', 'accuracy_new', 'accuracy_compare', 'accuracy_noise',
#     'accuracy_copy', 'accuracy_worst']
# legend = ['Train From Scratch', 'Enough New Data', 'Synthetic Data + Old Data', 'Synthetic Data', 'Old Data', 'Limited New Data + Noise',
#     'Limited New Data', 'Old Model Without Updating']
columns_for_plotting = ['accuracy_worst', 'accuracy_copy', 'accuracy_noise', 'accuracy_compare', 'accuracy_new', 'accuracy_combine',
    'accuracy_tf', 'accuracy_best']
legend = ['Old Model Without Updating', 'Limited New Data', 'Limited New Data + Noise', 'Old Data', 'Synthetic Data',
    'Synthetic Data + Old Data', 'Enough New Data', 'Train From Scratch']
title = 'Comparing Classification Accuracy of Different Model Training Methods'
Plot_Statistics.plotSubjectAdjacentTtest(mean_std_value, legend, columns_for_plotting, title=title, bonferroni_coeff=1)


##  plot confusion matrix
class_labels = ['LW', 'LW-SA', 'LW-SD', 'SA-LW', 'SA', 'SD-LW', 'SD']
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_best']['delay_0_ms'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_tf']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_combine']['delay_0_ms'], class_labels, normalize=False)
# # Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_new']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_compare']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_noise']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_worst']['delay_0_ms'], class_labels, normalize=False)




