##
from Bipolar_Recognition.Utility_Functions import Manipulate_Channels, Models
from Bipolar_EMG.Models import Dataset_Model
from Transition_Prediction.Models.ANN.Functions import Ann_Dataset


## read emg features
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']
feature_set = {'hdsemg_original': ['hdsemg_original', 'hdsemg_h_shift', 'hdsemg_v_shift'],
    'bipolar_original': ['bipolar_original', 'bipolar_h_shift_1', 'bipolar_h_shift_3', 'bipolar_h_shift_4', 'bipolar_h_shift_6',
        'bipolar_v_shift_1', 'bipolar_v_shift_3', 'bipolar_v_shift_4', 'bipolar_v_shift_6']}


for subject in subjects:
    for original_features in feature_set.keys():
        for shift_features in feature_set[original_features]:

            ## construct dataset
            emg_cross_validation = Manipulate_Channels.constructShiftDataset(subject, original_features, shift_features)
            # shuffle normalized dataset
            emg_normalized = Dataset_Model.combineNormalizedDataset(emg_cross_validation)
            emg_shuffled_groups = Ann_Dataset.shuffleTrainingSet(emg_normalized)

            ## classify using a single ann model
            models, model_results = Dataset_Model.classifyUsingAnnModel(emg_shuffled_groups)

            ## predict MV results
            predict_results, true_labels = Dataset_Model.reorganizePredictionResults(model_results)
            average_accuracies = []
            average_cm_numbers = []
            average_cm_recalls = []
            for mv_number in [5, 6, 7]:
                predict_mv_results = [{mode: Dataset_Model.majority_vote(value, n=mv_number) for mode, value in group_result.items()} for group_result in predict_results]
                average_accuracy, average_cm_number, average_cm_recall = Dataset_Model.calculateAccuracy(predict_mv_results, true_labels)
                average_accuracies.append(average_accuracy)
                average_cm_numbers.append(average_cm_number)
                average_cm_recalls.append(average_cm_recall)

            ## save results
            model_type = shift_features
            result_set = 0
            Models.saveResult(subject, average_accuracies, average_cm_numbers, average_cm_recalls, model_type, result_set, project='HDsEMG_Recognition')


