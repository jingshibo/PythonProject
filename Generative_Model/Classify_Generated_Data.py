##
from Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Generative_Model.Functions import Classify_Testing, Model_Storage, Data_Processing
import numpy as np
import datetime


## load pretrained classify model
subject = 'Number4'
version = 0  # the data from which experiment version to process
model_type = 'Raw_Cnn2d'  # Note: it requires reordering=True when train this model, in order to match the order of gan-generated data
model_name = list(range(5))
classify_models = Model_Storage.loadModels(subject, version, model_type, model_name, project='Insole_Emg')


## load rael data
#  define windows
down_sampling = True
start_before_toeoff_ms = 450
endtime_after_toeoff_ms = 400
feature_window_ms = 350
predict_window_ms = start_before_toeoff_ms
sample_rate = 1 if down_sampling is True else 2
predict_window_size = predict_window_ms * sample_rate
feature_window_size = feature_window_ms * sample_rate
predict_window_increment_ms = 20
feature_window_increment_ms = 20
predict_window_shift_unit = int(predict_window_increment_ms / feature_window_increment_ms)
predict_using_window_number = int((predict_window_size - feature_window_size) / (feature_window_increment_ms * sample_rate)) + 1
predict_window_per_repetition = int((endtime_after_toeoff_ms + start_before_toeoff_ms - predict_window_ms) / predict_window_increment_ms) + 1


## new data path
subject = 'Number5'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
# up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# down_up_session = [0, 1, 2, 3, 4, 5, 6, 8, 9]
up_down_session = [5, 6, 7, 8, 9]
down_up_session = [4, 5, 6, 8, 9]
sessions = [up_down_session, down_up_session]


## read and filter new data
split_parameters = Preprocessing.readSplitParameters(subject, version)
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters,
    start_position=-int(start_before_toeoff_ms * (2 / sample_rate)), end_position=int(endtime_after_toeoff_ms * (2 / sample_rate)),
    notchEMG=False, reordering=True, median_filtering=True)  # median filtering is necessary to avoid all zero values in a channel
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
del emg_filtered_data
real_emg_images = {k: [np.transpose(np.reshape(arr, newshape=(-1, 13, 10, 1), order='F'), (0, 3, 1, 2)).astype(np.float32) for arr in v] for
    k, v in emg_preprocessed.items()}
del emg_preprocessed


## generate fake data
# load gan models
subject = 'Test'
version = 1  # the data from which experiment version to process
model_type = 'CycleGAN'
model_name = ['gen_AB', 'gen_BA', 'disc_A', 'disc_B']
gan_models = Model_Storage.loadModels(subject, version, model_type, model_name, project='Generative_Model')
fake_old_emg = Data_Processing.generateFakeEmg(gan_models, real_emg_images, start_before_toeoff_ms, endtime_after_toeoff_ms)
fold = 5  # 5-fold cross validation
cross_validation_groups = Data_Preparation.crossValidationSet(fold, fake_old_emg)
del fake_old_emg


## process generated data
now = datetime.datetime.now()
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(cross_validation_groups, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
del cross_validation_groups
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups
print(datetime.datetime.now() - now)


## classify generated data
batch_size = 1024
now = datetime.datetime.now()
model_results = []
for number, model in classify_models.items():
    test_model = Classify_Testing.ModelTesting(model, batch_size)  # select one pretrained model for classifying data
    model_result = test_model.testModel(shuffled_groups)
    model_results.append(model_result)
print(datetime.datetime.now() - now)


## save model results
model_type = 'Raw_Cnn2d'
window_parameters = {'predict_window_ms': predict_window_ms, 'feature_window_ms': feature_window_ms, 'sample_rate': sample_rate,
    'predict_window_increment_ms': predict_window_increment_ms, 'feature_window_increment_ms': feature_window_increment_ms,
    'predict_window_shift_unit': predict_window_shift_unit, 'predict_using_window_number': predict_using_window_number,
    'endtime_after_toeoff_ms': endtime_after_toeoff_ms, 'predict_window_per_repetition': predict_window_per_repetition,
    'feature_window_per_repetition': feature_window_per_repetition}
for result_set in range(fold):
    Sliding_Ann_Results.saveModelResults(subject, model_results[result_set], version, result_set, window_parameters, model_type, project='Generative_Model')


## majority vote results using prior information, with a sliding windows to get predict results at different delay points
overall_accuracy = []
overall_cm_recall = []
for model_result in model_results:
    reorganized_results = MV_Results_ByGroup.groupedModelResults(model_result)
    sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results, feature_window_per_repetition,
        predict_window_shift_unit, predict_using_window_number, initial_start=0)
    accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
    # calculate the accuracy and cm. Note: the first dimension refers to each delay
    average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
        cm_bygroup)
    accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
        predict_window_shift_unit)
    overall_accuracy.append(accuracy)
    overall_cm_recall.append(cm_recall)
average_accuracy, average_cm_recall = Data_Processing.getAverageResults(overall_accuracy, overall_cm_recall)

