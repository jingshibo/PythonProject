'''calculate hdsemg classification performance under electrode shfit with the help of data augmentation'''

##
from HDsEMG_Recognition.Utility_Functions import Emg_Preprocessing, Manipulate_Channels, Model_Results
from Bipolar_EMG.Models import Dataset_Model
from Transition_Prediction.Models.ANN.Functions import Ann_Dataset
import gc


##
subjects = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Number6', 'Number7', 'Number8', 'Number9']
feature_set = ['left', 'up']


for subject in subjects:
    for test_shift_direction in feature_set:
        ## load interpolated emg features
        interp_emg_features = Emg_Preprocessing.readInterpFeatures(subject, 'hdsemg')

        ## construct augmented cross validation dataset
        random_shift_direction = 'both'  # shift electrode randomly on both horizontal and vertical direction
        original_emg, clip_shift_emg = Manipulate_Channels.clipShiftimages(interp_emg_features, shift_direction=random_shift_direction,
            max_shift=8, num_shifts=3, vertical_channels=slice(8, 89), horizontal_channels=slice(0, 25))
        del interp_emg_features
        gc.collect()
        # combine the shifted data into the training set for augmentation
        emg_cross_validation = Manipulate_Channels.constructAugmentDataset(original_emg, clip_shift_emg, test_shift_direction)
        del original_emg, clip_shift_emg
        gc.collect()

        ## shuffle normalized dataset
        emg_normalized = Dataset_Model.combineNormalizedDataset(emg_cross_validation)
        del emg_cross_validation
        gc.collect()
        emg_shuffled_groups = Ann_Dataset.shuffleTrainingSet(emg_normalized)
        del emg_normalized
        gc.collect()

        ## classify using a single ann model
        models, model_results = Model_Results.classifyUsingCnnModel(emg_shuffled_groups)
        del emg_shuffled_groups, models
        gc.collect()

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
        model_type = 'aug_both_4_' + test_shift_direction
        result_set = 0
        Model_Results.saveResult(subject, average_accuracies, average_cm_numbers, average_cm_recalls, model_type, result_set, project='HDsEMG_Recognition')

        # import winsound
        # frequency = 2500  # Set Frequency To 2500 Hertz
        # duration = 1500   # Set Duration To 1000 ms == 1 second
        # winsound.Beep(frequency, duration)

