'''
get oll subject's accuracy and confusion matrix at each delay point. display and plot
'''

## import
from Model_Raw.ConvRNN.Functions import Raw_ConvRnn_Results
from Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt

all_subjects = {}  # save all subject results
model_type = 'raw_ConvRnn'

## get model results. Note: different from the GRU method, here you can decide the shift_unit as you want.
subject = 'Shibo'
version = 1
result_set = 0
subject_results = Raw_ConvRnn_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Zehao'
version = 0
result_set = 0
subject_results = Raw_ConvRnn_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number1'
version = 0
result_set = 0
subject_results = Raw_ConvRnn_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number2'
version = 0
result_set = 0
subject_results = Raw_ConvRnn_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number3'
version = 0
result_set = 0
subject_results = Raw_ConvRnn_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number4'
version = 0
result_set = 0
subject_results = Raw_ConvRnn_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number5'
version = 0
result_set = 0
subject_results = Raw_ConvRnn_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number6'
version = 0
result_set = 0
subject_results = Raw_ConvRnn_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number7'
version = 0
result_set = 0
subject_results = Raw_ConvRnn_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number8'
version = 0
result_set = 0
subject_results = Raw_ConvRnn_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results


##  print accuracy results
subject = 'Shibo'
accuracy = all_subjects[f'{subject}']['accuracy']
x_label = [i * 32 for i in range(len(accuracy))]
y_value = [round(i, 1) for i in list(accuracy.values())]

fig, ax = plt.subplots()
bars = ax.bar(range(len(accuracy)), y_value)
ax.set_xticks(range(len(accuracy)), x_label)
ax.bar_label(bars)
ax.set_ylim([70, 100])
ax.set_xlabel('prediction time delay(ms)')
ax.set_ylabel('prediction accuracy(%)')
plt.title('Prediction accuracy for the prediction at different delay points')


##  print confusion matrix
subject = 'Shibo'
delay = 480
cm_call = all_subjects[f'{subject}']['cm_call'][f'delay_{delay}_ms']
class_labels = ['LW-LW', 'LW-SA', 'LW-SD', 'LW-SS', 'SA-LW', 'SA-SA', 'SA-SS', 'SD-LW', 'SD-SD', 'SD-SS', 'SS-LW', 'SS-SA', 'SS-SD']
Confusion_Matrix.plotConfusionMatrix(cm_call, class_labels, normalize=False)