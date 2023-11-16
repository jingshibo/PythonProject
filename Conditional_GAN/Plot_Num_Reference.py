'''
    plot the results of using different number of references for all subjects.
'''


##
from Conditional_GAN.Results import Load_Results, Subject_Result_Analysis, Num_Reference_Result_Analysis, Plot_Statistics
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt


##
all_subjects = {}
version = 0
filter_result_set = 2  # corresponds to the filter size (ten times): e.g., filter size = 10 relates to the results_set = 1
# num_reference = [0, 1, 2, 3, 5, 10]
num_reference = [0, 1, 3, 5]

##
subject = 'Number0'
all_subjects[subject] = Load_Results.getNumOfReferenceResults(subject, version, filter_result_set, num_reference)

##
subject = 'Number1'
all_subjects[subject] = Load_Results.getNumOfReferenceResults(subject, version, filter_result_set, num_reference)

##
subject = 'Number2'
all_subjects[subject] = Load_Results.getNumOfReferenceResults(subject, version, filter_result_set, num_reference)

# ##
# subject = 'Number3'
# all_subjects[subject] = Load_Results.getNumOfReferenceResults(subject, version, result_set, num_reference)

##
subject = 'Number4'
all_subjects[subject] = Load_Results.getNumOfReferenceResults(subject, version, filter_result_set, num_reference)

##
subject = 'Number5'
all_subjects[subject] = Load_Results.getNumOfReferenceResults(subject, version, filter_result_set, num_reference)

##
subject = 'Number6'
all_subjects[subject] = Load_Results.getNumOfReferenceResults(subject, version, filter_result_set, num_reference)

##
subject = 'Number7'
all_subjects[subject] = Load_Results.getNumOfReferenceResults(subject, version, filter_result_set, num_reference)

##
subject = 'Number8'
all_subjects[subject] = Load_Results.getNumOfReferenceResults(subject, version, filter_result_set, num_reference)


##
combined_results = Num_Reference_Result_Analysis.combineNumOfReferenceResults(all_subjects)
##
reorganized_results = Num_Reference_Result_Analysis.convertToDataframes(combined_results)
##
mean_std_value = Num_Reference_Result_Analysis.calcuNumOfReferenceStatValues(reorganized_results)


## calcualte statistic values
columns_for_plotting = ['accuracy_combine', 'accuracy_new', 'accuracy_compare', 'accuracy_noise', 'accuracy_copy']
legend = ['Synthetic Data + Old Data', 'Synthetic Data', 'Old Data', 'Limited New Data + Noise', 'Limited New Data']
# columns_for_plotting = ['accuracy_copy', 'accuracy_noise', 'accuracy_compare', 'accuracy_new', 'accuracy_combine']
# legend = ['Limited New Data', 'Limited New Data + Noise', 'Old Data', 'Synthetic Data + Old Data', 'Synthetic Data']
title = 'Comparing Classification Accuracy of Different Methods for Model Updating Using Different Number of Real Data'
Plot_Statistics.plotNumOfReferenceAdjacentTtest(mean_std_value, legend, columns_for_plotting, title=title, bonferroni_coeff=1)


# ##  plot confusion matrix
# class_labels = ['LW', 'LW-SA', 'LW-SD', 'SA-LW', 'SA', 'SD-LW', 'SD']
# reference = 1
# # Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall'][f'reference_{reference}']['cm_recall_combine'], class_labels, normalize=False)
# # Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall'][f'reference_{reference}']['cm_recall_new'], class_labels, normalize=False)
# # Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall'][f'reference_{reference}']['cm_recall_compare'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall'][f'reference_{reference}']['cm_recall_noise'], class_labels, normalize=False)
# Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall'][f'reference_{reference}']['cm_recall_copy'], class_labels, normalize=False)

