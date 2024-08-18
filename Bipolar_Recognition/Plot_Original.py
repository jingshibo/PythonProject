##
from Bipolar_Recognition.Utility_Functions import Model_Results
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix


## load results
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']
feature_set = ['hdsemg', 'bipolar_1', 'bipolar_2', 'bipolar_3', 'bipolar_4', 'bipolar_5', 'bipolar_6', 'bipolar_7', 'bipolar_8', 'bipolar_9']
result_set = 0

all_subjects = {}  # save all subject results
for subject in subjects:
    subject_results = {}
    for model_type in feature_set:
        subject_results[f'{model_type}_{result_set}'] = Model_Results.loadResult(subject, model_type, result_set, project='HDsEMG_Recognition')
    all_subjects[subject] = subject_results


## plot HDsEMG results
# plot HDsEMG accuracy for each subject (n=5,6,7)
hdsemg_results, hdsemg_accuracy, hdsemg_cm_recall = Model_Results.averageHdsemgResults(all_subjects)
Model_Results.plotHdsemgAccuracy(hdsemg_accuracy)

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
Model_Results.plotDerivedBipolarBox(bipolar_accuracy)





