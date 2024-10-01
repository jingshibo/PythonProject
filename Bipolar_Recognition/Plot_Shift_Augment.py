''' plot classification results under electrode shift and data augmentation'''

##
from Bipolar_Recognition.Utility_Functions import Model_Results


''' plot bipolar shift results '''
## load shift results
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number9']
shift_feature_set = ['hdsemg_original', 'hdsemg_h_shift', 'hdsemg_v_shift', 'bipolar_original', 'bipolar_h_shift_1', 'bipolar_h_shift_3',
    'bipolar_h_shift_4', 'bipolar_h_shift_6', 'bipolar_v_shift_1', 'bipolar_v_shift_3', 'bipolar_v_shift_4', 'bipolar_v_shift_6']
augment_feature_set = ['aug_both_left', 'aug_both_up']  # results after data augmentation
# augment_feature_set = ['aug_left', 'aug_up']  # results after data augmentation

# Combine the feature sets
all_feature_set = shift_feature_set + augment_feature_set
result_set = 0

all_subjects = {}  # save all subject results
for subject in subjects:
    subject_results = {}
    for model_type in all_feature_set:
        subject_results[f'{model_type}_{result_set}'] = Model_Results.loadResult(subject, model_type, result_set, project='HDsEMG_Recognition')
    all_subjects[subject] = subject_results

## extract shift accuracy values
shift_accuracy = {
    subject_key: {
        condition_key: {
            'accuracy': condition_data['accuracy'][0]
        } for condition_key, condition_data in subject_data.items()
    } for subject_key, subject_data in all_subjects.items()
}

## plot only bipolar shift results
shift_mean_std = Model_Results.calcuShiftMeanStd(shift_accuracy)
# Model_Results.plotShiftBipolarAccuracy(shift_mean_std)

## plot only HDsEMG shift results
# Model_Results.plotShiftHdsemgAccuracy(shift_mean_std)

## plot both bipolar EMG and HDsEMG shift results in a single chart
Model_Results.plotElectrodeShiftResults(shift_mean_std)





