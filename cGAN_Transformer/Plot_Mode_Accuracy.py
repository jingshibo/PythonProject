'''
    Extract the accuracy of only the transition modes from the confusion matrix for improvement comparison.
'''


##
from cGAN_Transformer.Functions import Result_Analysis
from Conditional_GAN.Results import Mode_Accuracy_Result_Analysis, Num_Reference_Result_Analysis, Plot_Statistics


##
all_subjects = {}
version = 0
result_set = 0
num_references = [3, 2, 1, 0]
gen_model = 'two_factors'


##
subject = 'Number0'
all_subjects[subject] = Result_Analysis.getModeAccuracyResults(subject, version, result_set, gen_model, num_references)
subject = 'Number1'
all_subjects[subject] = Result_Analysis.getModeAccuracyResults(subject, version, result_set, gen_model, num_references)
subject = 'Number2'
all_subjects[subject] = Result_Analysis.getModeAccuracyResults(subject, version, result_set, gen_model, num_references)
subject = 'Number3'
all_subjects[subject] = Result_Analysis.getModeAccuracyResults(subject, version, result_set, gen_model, num_references)
subject = 'Number4'
all_subjects[subject] = Result_Analysis.getModeAccuracyResults(subject, version, result_set, gen_model, num_references)
# subject = 'Number5'
# all_subjects[subject] = Result_Analysis.getModeAccuracyResults(subject, version, result_set, gen_model, num_references)
# subject = 'Number6'
# all_subjects[subject] = Result_Analysis.getModeAccuracyResults(subject, version, result_set, gen_model, num_references)
subject = 'Number7'
all_subjects[subject] = Result_Analysis.getModeAccuracyResults(subject, version, result_set, gen_model, num_references)
subject = 'Number8'
all_subjects[subject] = Result_Analysis.getModeAccuracyResults(subject, version, result_set, gen_model, num_references)


## combine the results
combined_results = Result_Analysis.combineSubjectResults(all_subjects)

## extract mode accuraies
extracted_mode_accuracy = Result_Analysis.extractModeAccuracyFromCm(combined_results)

## reorganize the results
reorganized_results = Num_Reference_Result_Analysis.convertToDataframes(extracted_mode_accuracy)

## calcualte statistic values
mean_std_value = Mode_Accuracy_Result_Analysis.CalcuModeAccuracyStataValues(reorganized_results)


## plot accuracy
columns_for_plotting = ['cm_recall_tf', 'cm_recall_mix_3', 'cm_recall_mix_2', 'cm_recall_mix_1', 'cm_recall_mix_0', 'cm_recall_worst']
legend = ['10 New Transition Data', '5 New Transition Data', '3 New Transition Data', '1 New Transition Data', '0 New Transition Data', 'No Model Updating']
title = ''
Plot_Statistics.plotModeAccuracyAdjacentTtest(mean_std_value, legend, columns_for_plotting, title=title, bonferroni_coeff=1)