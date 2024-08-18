##
from Bipolar_Recognition.Utility_Functions import Model_Results


''' plot bipolar shift results '''
## load shift results
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']
feature_set = ['hdsemg_original', 'hdsemg_h_shift', 'hdsemg_v_shift', 'bipolar_original', 'bipolar_h_shift_1', 'bipolar_h_shift_3',
    'bipolar_h_shift_4', 'bipolar_h_shift_6', 'bipolar_v_shift_1', 'bipolar_v_shift_3', 'bipolar_v_shift_4', 'bipolar_v_shift_6']
result_set = 0

subjects_shift = {}  # save all subject results
for subject in subjects:
    subject_results = {}
    for model_type in feature_set:
        subject_results[f'{model_type}_{result_set}'] = Model_Results.loadResult(subject, model_type, result_set, project='HDsEMG_Recognition')
    subjects_shift[subject] = subject_results

# extract shift accuracy values
shift_accuracy = {
    subject_key: {
        condition_key: {
            'accuracy': condition_data['accuracy'][0]
        } for condition_key, condition_data in subject_data.items()
    } for subject_key, subject_data in subjects_shift.items()
}

# plot only bipolar shift results
shift_mean_std = Model_Results.calcuShiftMeanStd(shift_accuracy)
Model_Results.plotShiftBipolarAccuracy(shift_mean_std)


''' plot HDsEMG shift results '''
## load augmentation results
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']
feature_set = ['aug_left', 'aug_up']
result_set = 0

subjects_augment = {}  # save all subject results
for subject in subjects:
    subject_results = {}
    for model_type in feature_set:
        subject_results[f'{model_type}_{result_set}'] = Model_Results.loadResult(subject, model_type, result_set, project='HDsEMG_Recognition')
    subjects_augment[subject] = subject_results

augment_accuracy = {  # extract augment accuracy values
    subject_key: {
        condition_key: {
            'accuracy': condition_data['accuracy'][0]
        } for condition_key, condition_data in subject_data.items()
    } for subject_key, subject_data in subjects_augment.items()
}

# plot only HDsEMG shift results
aug_mean_std = Model_Results.calcuShiftMeanStd(subjects_augment)
shift_mean_std.update(aug_mean_std)  # combine the content of aug_mean_std into previous dict shift_mean_std
Model_Results.plotShiftHdsemgAccuracy(shift_mean_std)


''' plot both bipolar EMG and HDsEMG shift results '''
##
Model_Results.plotElectrodeShiftResults(shift_mean_std)





