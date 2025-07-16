##
from cGAN_Transformer.Functions import Result_Analysis
from Conditional_GAN.Results import Plot_Statistics
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix


##
all_subjects = {}
version = 0
result_set = 0
num_reference = 0
gen_model = 'two_factors'  # one_factor or two_factors


##
subject = 'Number0'
all_subjects[subject] = Result_Analysis.getOldResults(subject, version, gen_model, result_set)
subject = 'Number1'
all_subjects[subject] = Result_Analysis.getOldResults(subject, version, gen_model, result_set)
subject = 'Number2'
all_subjects[subject] = Result_Analysis.getOldResults(subject, version, gen_model, result_set)
subject = 'Number3'
all_subjects[subject] = Result_Analysis.getOldResults(subject, version, gen_model, result_set)
subject = 'Number4'
all_subjects[subject] = Result_Analysis.getOldResults(subject, version, gen_model, result_set)
subject = 'Number5'
all_subjects[subject] = Result_Analysis.getOldResults(subject, version, gen_model, result_set)
# subject = 'Number6'
# all_subjects[subject] = Results.getOldResults(subject, version, gen_model, result_set)
subject = 'Number7'
all_subjects[subject] = Result_Analysis.getOldResults(subject, version, gen_model, result_set)
# subject = 'Number8'
# all_subjects[subject] = Results.getOldResults(subject, version, gen_model, result_set)


## combine the results
combined_results = Result_Analysis.combineSubjectResults(all_subjects)


## calcualte statistic values
mean_std_value = Result_Analysis.calcuSubjectStatValues(combined_results)


## plot old model results
columns_for_plotting = ['accuracy_old_real', 'accuracy_old_rebalanced', 'accuracy_old_synthetic', 'accuracy_old_imbalance', 'accuracy_old_noisy', 'accuracy_old_copy']
legend = ['Dataset 2', 'Dataset 4', 'Dataset 1', 'Dataset 3', 'Dataset 6', 'Dataset 5']
title = ''
Plot_Statistics.plotSubjectAdjacentTtest(mean_std_value, legend, columns_for_plotting, title=title, bonferroni_coeff=1)


##  plot old model confusion matrix
class_labels = ['LW', 'LWSA', 'LWSD', 'SALW', 'SA', 'SDLW', 'SD']
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_old_real'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_old_synthetic'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_old_imbalance'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_old_rebalanced'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_old_copy'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['cm_recall_old_noisy'], class_labels, normalize=False)
