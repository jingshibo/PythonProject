##
from Bipolar_Recognition.Utility_Functions import Emg_Preprocessing
from Bipolar_EMG.Models import Dataset_Model
from Bipolar_Recognition.Models import Result_Processing
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Transition_Prediction.Models.ANN.Functions import Ann_Dataset
import datetime


## read emg features
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']
feature_set = ['hdsemg', 'bipolar_1', 'bipolar_2', 'bipolar_3', 'bipolar_4', 'bipolar_5', 'bipolar_6', 'bipolar_7', 'bipolar_8', 'bipolar_9']

for subject in subjects:
    for feature_name in feature_set:
        emg_features = Emg_Preprocessing.readFeatures(subject, feature_name)
        emg_cross_validation = Data_Preparation.crossValidationSet(5, emg_features, shuffle=False)

        ## shuffle normalized dataset
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
        model_type = feature_name
        result_set = 0
        Result_Processing.saveResult(subject, average_accuracies, average_cm_numbers, average_cm_recalls, model_type, result_set, project='HDsEMG_Recognition')


        # ## load results
        # classify_results = Result_Processing.loadResult(subject, model_type, result_set, project='HDsEMG_Recognition')