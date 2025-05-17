''' plot classification results without electrode shift'''

##
from HDsEMG_Recognition.Utility_Functions import Model_Results
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix


## load results
# subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']  # bipolar plot
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7']  # hdsemg plot
feature_set = ['hdsemg', 'LDA', 'QDA', 'RF', 'SVM_linear', 'SVM_rbf', 'bipolar_1', 'bipolar_2', 'bipolar_3', 'bipolar_4', 'bipolar_5',
    'bipolar_6', 'bipolar_7', 'bipolar_8', 'bipolar_9']
result_set = 0

all_subjects = {}  # save all subject results
for subject in subjects:
    subject_results = {}
    for model_type in feature_set:
        subject_results[f'{model_type}_{result_set}'] = Model_Results.loadResult(subject, model_type, result_set, project='HDsEMG_Recognition')
    all_subjects[subject] = subject_results


## plot HDsEMG classification accuracy
model_accuracy = Model_Results.reorganize_by_model(all_subjects)
model_accuracy_statistics = Model_Results.compute_model_accuracy_stats(model_accuracy)
Model_Results.plotModelAccuracy(model_accuracy_statistics)

# plot HDsEMG accuracy for each subject (n=5,6,7)
hdsemg_results, hdsemg_accuracy, hdsemg_cm_recall = Model_Results.averageHdsemgResults(all_subjects)
# Model_Results.plotHdsemgAccuracy(hdsemg_accuracy)

# plot HDsEMG confusion matrix when n=5
cm_call = hdsemg_cm_recall['average'][0]
class_labels = ['LW', 'RA', 'RD', 'SA', 'SD', 'SS']
Confusion_Matrix.plotConfusionMatrix(cm_call, class_labels, normalize=False)


## plot bipolar results
# select only the accuracy values when n=5
bipolar_accuracy = {
    subject_key: {
        condition_key: {
            'accuracy': condition_data['accuracy'][0]
        } for condition_key, condition_data in subject_data.items()
    } for subject_key, subject_data in all_subjects.items()
}
median_accuracy = Model_Results.plotDerivedBipolarBox(bipolar_accuracy)
mean_accuracy = Model_Results.averageBipolarAccuracies(bipolar_accuracy)


##
import matplotlib.pyplot as plt
import numpy as np


# First dataset: 10 groups
data1 = [np.random.normal(loc=0, scale=1, size=100) for _ in range(10)]
# Second dataset: only first 6 groups
data2 = [np.random.normal(loc=1, scale=1, size=100) for _ in range(6)]

# Positions
positions1 = [i * 2 for i in range(10)]             # 0, 2, 4, ..., 18
positions2 = [i * 2 + 1 for i in range(6)]          # 1, 3, 5, ..., 11

# Plot
plt.boxplot(data1, positions=positions1, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor="skyblue"))
plt.boxplot(data2, positions=positions2, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor="lightgreen"))

# Create x-ticks centered for each x group
xtick_positions = positions1[:6]  # x-ticks for shared data
xtick_positions.extend(positions1[6:])  # x-ticks only for data1

xtick_labels = [f'Group {i+1}' for i in range(10)]
plt.xticks(positions1, xtick_labels)

# Labels and legend
plt.xlabel('Groups')
plt.ylabel('Values')
plt.title('Mixed Boxplot: 10 vs. 6 X-Ticks')
plt.legend(['Data1 (All)', 'Data2 (Partial)'], loc='upper right')

plt.tight_layout()
plt.show()
