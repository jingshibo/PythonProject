'''
get oll subject's accuracy and confusion matrix at each delay point. display and plot
'''

## import
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt
all_subjects = {}  # save all subject results


##
model_type = 'raw_Cnn2d'
result_set = 1

## get model results. Note: different from the GRU method, here you can decide the shift_unit as you want.
subject = 'Shibo'
version = 1
subject_results = Sliding_Ann_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Zehao'
version = 0
subject_results = Sliding_Ann_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number1'
version = 0
subject_results = Sliding_Ann_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number2'
version = 0
subject_results = Sliding_Ann_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number3'
version = 0
subject_results = Sliding_Ann_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number4'
version = 0
subject_results = Sliding_Ann_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number5'
version = 0
subject_results = Sliding_Ann_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number6'
version = 0
subject_results = Sliding_Ann_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number7'
version = 0
subject_results = Sliding_Ann_Results.getPredictResults(subject, version, result_set, model_type)
all_subjects[subject] = subject_results
##
subject = 'Number8'
version = 0
subject_results = Sliding_Ann_Results.getPredictResults(subject, version, result_set, model_type)
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



##  print all accuracy results
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
accuracy = {'delay_0_ms': 91.4, 'delay_60_ms': 93.2, 'delay_100_ms': 95.1, 'delay_200_ms': 97.4, 'delay_300_ms': 98.8, 'delay_400_ms': 99.5}
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

