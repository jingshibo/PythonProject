'''
get oll subject's accuracy and confusion matrix at each delay point. display and plot
'''

## import
from Model_Sliding.GRU.Functions import Sliding_Gru_Model
from Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt

all_subjects = {}  # save all subject results
result_set = 0

## load model results
subject = 'Shibo'
version = 1
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set)
all_subjects[subject] = subject_results
##
subject = 'Zehao'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set)
all_subjects[subject] = subject_results
##
subject = 'Number1'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set)
all_subjects[subject] = subject_results
##
subject = 'Number2'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set)
all_subjects[subject] = subject_results
##
subject = 'Number3'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set)
all_subjects[subject] = subject_results
##
subject = 'Number4'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set)
all_subjects[subject] = subject_results
##
subject = 'Number5'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set)
all_subjects[subject] = subject_results
##
subject = 'Number6'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set)
all_subjects[subject] = subject_results
##
subject = 'Number7'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set)
all_subjects[subject] = subject_results
##
subject = 'Number8'
version = 0
subject_results = Sliding_Gru_Model.getPredictResults(subject, version, result_set)
all_subjects[subject] = subject_results


##  print accuracy results
subject = 'Zehao'
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
subject = 'Zehao'
delay = 480
cm_call = all_subjects[f'{subject}']['cm_call'][f'delay_{delay}_ms']
class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'SALW', 'SASA', 'SASS', 'SDLW', 'SDSD', 'SDSS', 'SSLW', 'SSSA', 'SSSD']
plt.figure()
Confusion_Matrix.plotConfusionMatrix(cm_call, class_labels, normalize=False)