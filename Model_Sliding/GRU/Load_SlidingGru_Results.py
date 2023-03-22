'''
get oll subject's accuracy and confusion matrix at each delay point. display and plot
'''


## import
from Model_Sliding.GRU.Functions import Sliding_Gru_Model
from Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt

all_subjects = {}  # save all subject results
result_set = 0  # default for sliding GRU model results
model_type = 'sliding_GRU'

## load model results
subject = 'Shibo'
version = 1
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Zehao'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number1'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number2'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number3'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number4'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number5'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number6'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number7'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number8'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results


##  average accuracy across subjects
delay_groups = list(all_subjects['Shibo']['accuracy'].keys())  # list all transition types
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


##  print confusion matrix
delay = 100
cm_call = average_cm[f'delay_{delay}_ms']
# subject = 'Shibo'
# cm_call = all_subjects[f'{subject}']['cm_call'][f'delay_{delay}_ms']
class_labels = ['LW-LW', 'LW-SA', 'LW-SD', 'LW-SS', 'SA-LW', 'SA-SA', 'SA-SS', 'SD-LW', 'SD-SD', 'SD-SS', 'SS-LW', 'SS-SA', 'SS-SD']
plt.figure()
Confusion_Matrix.plotConfusionMatrix(cm_call, class_labels, normalize=False)


##
# import numpy as np
# import matplotlib.pyplot as plt
# cm = np.array([[96,	0,	3,	0,	0,	0,	0,	0,	0,	2,	0,	0,	0],
# [0,	95,	1,	4,	0,	1,	0,	0,	0,	0,	2,	0,	0],
# [2,	0,	94,	2,	1,	0,	0,	0,	0,	0,	0,	1,	0],
# [2,	1,	2,	92,	0,	0,	0,	0,	2,	0,	0,	0,	0],
# [0,	0,	0,	0,	94,	0,	1,	0,	0,	0,	0,	3,	0],
# [0,	0,	0,	0,	0,	92,	1,	1,	0,	0,	0,	0,	10],
# [0,	0,	0,	0,	0,	1,	95,	0,	1,	0,	0,	0,	1],
# [0,	1,	0,	0,	2,	1,	0,	93,	0,	0,	0,	0,	0],
# [0,	2,	0,	1,	0,	0,	1,	1,	96,	0,	3,	0,	0],
# [0,	0,	0,	0,	0,	0,	1,	0,	0,	98,	0,	0,	0],
# [0,	1,	0,	0,	1,	1,	1,	4,	0,	0,	95,	0,	0],
# [0,	0,	0,	0,	2,	0,	0,	0,	0,	0,	0,	96,	1],
# [0,	0,	0,	1,	0,	4,	0,	1,	1,	0,	0,	0,	88]])
# class_labels = ['LW-SA', 'LW-SD', 'LW-RA', 'LW-RD', 'RA-LW', 'RD-LW', 'SA-LW', 'SD-LW', 'LW-LW', 'SA-SA', 'SD-SD', 'RA-RA', 'RD-RD']
# plt.figure()
# Confusion_Matrix.plotConfusionMatrix(cm, class_labels, normalize=True)
#
# ##
# import matplotlib.pyplot as plt
# from Models.Functions import Confusion_Matrix
# recall = cm_recall['delay_360_ms']
# class_labels = ['LW-LW', 'LW-SA', 'LW-SD', 'LW-SS', 'SA-LW', 'SA-SA', 'SA-SS', 'SD-LW', 'SD-SD', 'SD-SS', 'SS-LW', 'SS-SA', 'SS-SD']
# plt.figure()
# Confusion_Matrix.plotConfusionMatrix(recall, class_labels, normalize=False)
