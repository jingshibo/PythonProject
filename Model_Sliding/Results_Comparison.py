## import
from Model_Sliding.Functions import Sliding_Gru_Model, Sliding_Evaluation_ByGroup
from Models.Utility_Functions import Confusion_Matrix
import matplotlib.pyplot as plt


## get all subject results
all_subjects = {}  # save subject results
def getSubjectResult(subject, result_set=0, version=0):
    model_results = Sliding_Gru_Model.loadModelResults(subject, version, result_set)
    delay_results = Sliding_Evaluation_ByGroup.getResultsByDelay(model_results)

    accuracy = {}
    cm_recall = {}
    for key, value in delay_results.items():
        accuracy[key] = value['overall_accuracy']
        cm_recall[key] = value['cm_recall']
    subject_results = {'accuracy': accuracy, 'cm_call': cm_recall}

    return subject_results


## load model results
subject = 'Shibo'
subject_results = getSubjectResult(subject)
all_subjects[subject] = subject_results
##
subject = 'Zehao'
subject_results = getSubjectResult(subject)
all_subjects[subject] = subject_results
##
subject = 'Number1'
subject_results = getSubjectResult(subject)
all_subjects[subject] = subject_results
##
subject = 'Number2'
subject_results = getSubjectResult(subject)
all_subjects[subject] = subject_results
##
subject = 'Number3'
subject_results = getSubjectResult(subject)
all_subjects[subject] = subject_results
##
subject = 'Number4'
subject_results = getSubjectResult(subject)
all_subjects[subject] = subject_results
##
subject = 'Number5'
subject_results = getSubjectResult(subject)
all_subjects[subject] = subject_results
##
subject = 'Number6'
subject_results = getSubjectResult(subject)
all_subjects[subject] = subject_results
##
subject = 'Number7'
subject_results = getSubjectResult(subject)
all_subjects[subject] = subject_results
##
subject = 'Number8'
subject_results = getSubjectResult(subject)
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