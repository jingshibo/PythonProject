'''
get oll subject's accuracy and confusion matrix at each delay point. display and plot
'''

## import
import copy

import pandas as pd

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


##  average accuracy across subjects
delay_groups = list(all_subjects['Number5']['accuracy'].keys())  # list all transition types
average_accuracy = {delay_time: 0 for delay_time in delay_groups}  # initialize average accuracy list
average_cm = {delay_time: 0 for delay_time in delay_groups}  # initialize average cm list

for subject_number, subject_results in all_subjects.items():
    for key, value in subject_results.items():
        for delay_time, delay_value in value.items():
            if key == 'accuracy':
                average_accuracy[delay_time] = average_accuracy[delay_time] + delay_value
            elif key == 'cm_call':
                average_cm[delay_time] = average_cm[delay_time] + delay_value

for delay_time, delay_value in average_accuracy.items():
    average_accuracy[delay_time] = delay_value / len(all_subjects)
for delay_time, delay_value in average_cm.items():
    average_cm[delay_time] = delay_value / len(all_subjects)


##  print accuracy results
# subject = 'Shibo'
# accuracy = all_subjects[f'{subject}']['accuracy']
accuracy = average_accuracy
predict_window_increment_ms = 20
x_label = [i * predict_window_increment_ms for i in range(len(accuracy))]
y_value = [round(i, 1) for i in list(accuracy.values())]

fig, ax = plt.subplots()
bars = ax.bar(range(len(accuracy)), y_value)
ax.set_xticks(range(len(accuracy)), x_label)
ax.bar_label(bars)
ax.set_ylim([90, 100])
ax.set_xlabel('prediction time delay(ms)')
ax.set_ylabel('prediction accuracy(%)')
plt.title('Prediction accuracy for the prediction at different delay points')


##  print accuracy results at each 100 delay
delay_list = ['delay_0_ms', 'delay_60_ms', 'delay_100_ms', 'delay_200_ms', 'delay_300_ms', 'delay_400_ms']
# accuracy = {delay: average_accuracy[delay] for delay in delay_list}
accuracy = {'delay_0_ms': 90.1, 'delay_60_ms': 92.0, 'delay_100_ms': 94.7, 'delay_200_ms': 96.3, 'delay_300_ms': 98.2, 'delay_400_ms': 99.3}
x_label = delay_list
y_value = [round(i, 1) for i in list(accuracy.values())]

fig, ax = plt.subplots()
bars = ax.bar(range(len(accuracy)), y_value)
ax.set_xticks(range(len(accuracy)), x_label)
ax.bar_label(bars)
ax.set_ylim([90, 100])
ax.set_xlabel('prediction time delay(ms)')
ax.set_ylabel('prediction accuracy(%)')
plt.title('Prediction accuracy for the prediction at different delay points')


##  print confusion matrix
delay = 100
cm_call = average_cm[f'delay_{delay}_ms']
# subject = 'Shibo'
# cm_call = all_subjects[f'{subject}']['cm_call'][f'delay_{delay}_ms']
class_labels = ['LW-LW', 'LW-SA', 'LW-SD', 'LW-SS', 'SA-LW', 'SA-SA', 'SA-SS', 'SD-LW', 'SD-SD', 'SD-SS', 'SS-LW', 'SS-SA', 'SS-SD']
plt.figure()
Confusion_Matrix.plotConfusionMatrix(cm_call, class_labels, normalize=False)


## extract the results from each transition mode
delay_groups = list(all_subjects['Number5']['accuracy'].keys())  # list all transition types
delay_list = ['delay_0_ms', 'delay_60_ms', 'delay_100_ms', 'delay_200_ms', 'delay_300_ms', 'delay_400_ms']

mean_list = []
cm_list = []
for i in range(13):
    total_accuracy = {delay_time: [] for delay_time in delay_groups if delay_time in delay_list}  # initialize average accuracy list
    total_cm = {delay_time: [] for delay_time in delay_groups if delay_time in delay_list}  # initialize average cm list

    for subject_number, subject_results in all_subjects.items():
        for key, value in subject_results.items():
            for delay_time, delay_value in value.items():
                if delay_time in ['delay_0_ms', 'delay_60_ms', 'delay_100_ms', 'delay_200_ms', 'delay_300_ms', 'delay_400_ms']:
                    if key == 'accuracy':
                        total_accuracy[delay_time].append(delay_value)
                    elif key == 'cm_call':
                        total_cm[delay_time].append(delay_value[i, i])  # only select the SDSS mode

    mean_cm = {delay_time: 0 for delay_time in delay_groups if delay_time in delay_list}
    std_cm = {delay_time: 0 for delay_time in delay_groups if delay_time in delay_list}
    for delay, value in total_cm.items():
        mean_cm[delay] = (pd.DataFrame(total_cm[delay])).mean()
        std_cm[delay] = (pd.DataFrame(total_cm[delay])).std()

    mean_list.append(mean_cm)
    cm_list.append(std_cm)

# plot the accuracy of a certain mode
mode = 9
mean_cm = pd.DataFrame(mean_list[mode]).T
std_cm = pd.DataFrame(cm_list[mode]).T
# ax = mean_cm.plot.bar()
ax = mean_cm.plot.bar(yerr=std_cm, capsize=4, width=0.8)
# Add values on top of the bars
for i in range(mean_cm.shape[0]):
    plt.text(i, mean_cm.iloc[i] + 0.01, str(round(mean_cm.iloc[i], 3).values), ha='center')


