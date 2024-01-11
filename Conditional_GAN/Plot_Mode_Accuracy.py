'''
    Extract the accuracy of only the transition modes from the confusion matrix for improvement comparison.
'''


##
from Conditional_GAN.Results import Load_Results, Subject_Result_Analysis, Mode_Accuracy_Result_Analysis, Num_Reference_Result_Analysis, Plot_Statistics
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt


##
all_subjects = {}
version = 0
basis_result_set = 0
filter_result_set = 2  # corresponds to the filter size (ten times): e.g., filter size = 10 relates to the results_set = 1
# num_reference = [0, 1, 2, 3, 5, 10]
num_reference = [5, 3, 1, 0]


##
subject = 'Number0'
all_subjects[subject] = Load_Results.getModeAccuracyResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number1'
all_subjects[subject] = Load_Results.getModeAccuracyResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number2'
all_subjects[subject] = Load_Results.getModeAccuracyResults(subject, version, basis_result_set, filter_result_set, num_reference)

# ##
# subject = 'Number3'
# all_subjects[subject] = Load_Results.getCombineDataResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number4'
all_subjects[subject] = Load_Results.getModeAccuracyResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number5'
all_subjects[subject] = Load_Results.getModeAccuracyResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number6'
all_subjects[subject] = Load_Results.getModeAccuracyResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number7'
all_subjects[subject] = Load_Results.getModeAccuracyResults(subject, version, basis_result_set, filter_result_set, num_reference)

##
subject = 'Number8'
all_subjects[subject] = Load_Results.getModeAccuracyResults(subject, version, basis_result_set, filter_result_set, num_reference)


## combine the results
combined_results = Subject_Result_Analysis.combineSubjectResults(all_subjects)

## extract mode accuraies
extracted_mode_accuracy = Mode_Accuracy_Result_Analysis.extractModeAccuracyFromCm(combined_results)

## reorganize the results
reorganized_results = Num_Reference_Result_Analysis.convertToDataframes(extracted_mode_accuracy)

## calcualte statistic values
mean_std_value = Mode_Accuracy_Result_Analysis.CalcuModeAccuracyStataValues(reorganized_results)


## plot accuracy
columns_for_plotting = ['cm_recall_tf', 'cm_recall_combine_5', 'cm_recall_combine_3', 'cm_recall_combine_1', 'cm_recall_combine_0', 'cm_recall_worst']
legend = ['10 New Transition Data', '5 New Transition Data', '3 New Transition Data', '1 New Transition Data', '0 New Transition Data', 'No Model Updating']
# columns_for_plotting = ['cm_recall_worst', 'cm_recall_combine_0', 'cm_recall_combine_1', 'cm_recall_combine_3', 'cm_recall_combine_5', 'cm_recall_tf']
# legend = ['No Updating', '0 Real Transition Data', '1 Real Transition Data', '3 Real Transition Data', '5 Real Transition Data', '10 Real Transition Data']
title = ''
Plot_Statistics.plotModeAccuracyAdjacentTtest(mean_std_value, legend, columns_for_plotting, title=title, bonferroni_coeff=1)

