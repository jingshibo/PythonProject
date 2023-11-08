##
from Conditional_GAN.Results import Result_Analysis, Plot_Statistics
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt


##
all_subjects = {}
version = 0
result_set = 1  # corresponds to the filter size (ten times): e.g., filter size = 10 relates to the results_set = 1
# num_reference = [0, 1, 2, 3, 5, 10]
num_reference = [0, 1, 3, 5]

##
subject = 'Number0'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, num_reference)

##
subject = 'Number1'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, num_reference)

##
subject = 'Number2'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, num_reference)

# ##
# subject = 'Number3'
# all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, num_reference)

##
subject = 'Number4'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, num_reference)

##
subject = 'Number5'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, num_reference)

##
subject = 'Number6'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, num_reference)

##
subject = 'Number7'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, num_reference)

##
subject = 'Number8'
all_subjects[subject] = Result_Analysis.getNumOfReferenceResults(subject, version, result_set, num_reference)


##
combined_results = Result_Analysis.combineNumOfReferenceResults(all_subjects)
##
reorganized_results = Result_Analysis.convertToDataframes(combined_results)
##
mean_std_value = Result_Analysis.calcuNumOfReferenceStatValues(reorganized_results)


## calcualte statistic values
columns_for_plotting = ['accuracy_combine', 'accuracy_new', 'accuracy_compare', 'accuracy_noise', 'accuracy_copy']
legend = ['Add Synthetic + Old Data', 'Add Synthetic Data', 'Add Old Data', 'Add Noise', 'No Augmentation']
# legend = ['All New Data', 'Add Synthetic + Old Data', 'Add Synthetic Data', 'Add Old Data', 'Add Noise', 'No Augmentation', 'No Update']
title = 'Comparing Classification Accuracy of Different Methods for Model Updating Using Different Number of Real Data'
Plot_Statistics.plotNumOfReferenceAdjacentTtest(mean_std_value, legend, columns_for_plotting, title=title, bonferroni_coeff=1)

##  plot confusion matrix
class_labels = ['LW-LW', 'LW-SA', 'LW-SD', 'SA-LW', 'SA-SA', 'SD-LW', 'SD-SD']
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['reference_0']['cm_recall_combine'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['reference_0']['cm_recall_new'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['reference_0']['cm_recall_compare'], class_labels, normalize=False)
Confusion_Matrix.plotConfusionMatrix(mean_std_value['cm_recall']['reference_0']['cm_recall_noise'], class_labels, normalize=False)

