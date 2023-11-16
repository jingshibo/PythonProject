'''
    plot the results of only old data.
'''


##
from Conditional_GAN.Results import Subject_Result_Analysis, Plot_Statistics
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt


##
all_subjects = {}
version = 0
basis_result_set = 0
old_result_set = 1  # corresponds to the filter size (ten times): e.g., filter size = 10 relates to the results_set = 1
# num_reference = [0, 1, 2, 3, 5, 10]
num_reference = [0, 1, 2, 3, 5, 10]


##
subject = 'Number0'
all_subjects[subject] = Subject_Result_Analysis.getOldDataResults(subject, version, basis_result_set, old_result_set, num_reference)

##
subject = 'Number1'
all_subjects[subject] = Subject_Result_Analysis.getOldDataResults(subject, version, basis_result_set, old_result_set, num_reference)

##
subject = 'Number2'
all_subjects[subject] = Subject_Result_Analysis.getOldDataResults(subject, version, basis_result_set, old_result_set, num_reference)

##
subject = 'Number3'
all_subjects[subject] = Subject_Result_Analysis.getOldDataResults(subject, version, basis_result_set, old_result_set, num_reference)

##
subject = 'Number4'
all_subjects[subject] = Subject_Result_Analysis.getOldDataResults(subject, version, basis_result_set, old_result_set, num_reference)

##
subject = 'Number5'
all_subjects[subject] = Subject_Result_Analysis.getOldDataResults(subject, version, basis_result_set, old_result_set, num_reference)

##
subject = 'Number6'
all_subjects[subject] = Subject_Result_Analysis.getOldDataResults(subject, version, basis_result_set, old_result_set, num_reference)

# ##
# subject = 'Number7'
# all_subjects[subject] = Result_Analysis.getOldDataResults(subject, version, basis_result_set, old_result_set, num_reference)
#
# ##
# subject = 'Number8'
# all_subjects[subject] = Result_Analysis.getOldDataResults(subject, version, basis_result_set, old_result_set, num_reference)


## combine the results
combined_results = Subject_Result_Analysis.combineSubjectResults(all_subjects)

## calcualte statistic values
mean_std_value = Subject_Result_Analysis.calcuSubjectStatValues(combined_results)


# ## plot accuracy
# columns_for_plotting = ['accuracy_best', 'accuracy_tf', 'accuracy_combine', 'accuracy_new', 'accuracy_compare', 'accuracy_noise',
#     'accuracy_copy', 'accuracy_worst']
# legend = ['All New Data', 'All New TF', 'Add Synthetic + Old Data', 'Add Synthetic Data', 'Add Old Data', 'Add Noise', 'No Augmentation', 'No Update']
# # legend = ['All New Data', 'Add Synthetic + Old Data', 'Add Synthetic Data', 'Add Old Data', 'Add Noise', 'No Augmentation', 'No Update']
# title = 'Comparing Classification Accuracy of Different Methods for Model Updating '
# Plot_Statistics.plotSubjectAdjacentTtest(mean_std_value, legend, columns_for_plotting, title=title, bonferroni_coeff=1)
#
#
# ##  plot confusion matrix
# class_labels = ['LW-LW', 'LW-SA', 'LW-SD', 'SA-LW', 'SA-SA', 'SD-LW', 'SD-SD']
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_best']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_tf']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_combine']['delay_0_ms'], class_labels, normalize=False)
# # Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_new']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_compare']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_noise']['delay_0_ms'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_worst']['delay_0_ms'], class_labels, normalize=False)