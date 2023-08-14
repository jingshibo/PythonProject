##
from Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Cycle_GAN.Functions import Classify_Testing, Model_Storage, Data_Processing, CycleGAN_Training, Visualization, CycleGAN_Testing
import numpy as np
import datetime


'''generate fake data'''
##  define windows
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


## old data
subject = 'Number4'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
# up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# down_up_session = [0, 1, 2, 5, 6, 7, 8, 9, 10]
up_down_session = [0, 1, 2, 3, 4]
down_up_session = [0, 1, 2, 5, 6]
sessions = [up_down_session, down_up_session]


## read and filter data
split_parameters = Preprocessing.readSplitParameters(subject, version)
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters,
    start_position=-int(start_before_toeoff_ms * (2 / sample_rate)), end_position=int(endtime_after_toeoff_ms * (2 / sample_rate)),
    notchEMG=False, reordering=True, median_filtering=True)  # median filtering is necessary to avoid all zero values in a channel
old_emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
del emg_filtered_data


## new data
subject = 'Number5'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
# up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# down_up_session = [0, 1, 2, 3, 4, 5, 6, 8, 9]
up_down_session = [5, 6, 7, 8, 9]
down_up_session = [4, 5, 6, 8, 9]
sessions = [up_down_session, down_up_session]


## read and filter data
split_parameters = Preprocessing.readSplitParameters(subject, version)
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters,
    start_position=-int(start_before_toeoff_ms * (2 / sample_rate)), end_position=int(endtime_after_toeoff_ms * (2 / sample_rate)),
    notchEMG=False, reordering=True, median_filtering=True)  # median filtering is necessary to avoid all zero values in a channel
new_emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
del emg_filtered_data


## visualize data distribution
# Visualization.plotHistPercentage(old_emg_images)
# Visualization.plotHistByClass(old_emg_images)


## clip the values and normalize data
limit = 1500
old_emg_reshaped = {k: [np.transpose(np.reshape(arr, newshape=(-1, 13, 10, 1), order='F'), (0, 3, 1, 2)).astype(np.float32) for arr in v] for
    k, v in old_emg_preprocessed.items()}
# del old_emg_preprocessed
new_emg_reshaped = {k: [np.transpose(np.reshape(arr, newshape=(-1, 13, 10, 1), order='F'), (0, 3, 1, 2)).astype(np.float32) for arr in v] for
    k, v in new_emg_preprocessed.items()}
# del new_emg_preprocessed
old_emg_normalized = Data_Processing.normalizeEmgData(old_emg_reshaped, limit=limit)
new_emg_normalized = Data_Processing.normalizeEmgData(new_emg_reshaped, limit=limit)


##  train generative models
# model data
subject = 'Test'
version = 1  # the data from which experiment version to process
old_LWLW_data = old_emg_normalized['emg_LWLW']
new_LWLW_data = new_emg_normalized['emg_LWLW']
sample_number = min(len(old_LWLW_data), len(new_LWLW_data))

# hyperparameters
num_epochs = 400  # the number of times you iterate through the entire dataset when training
decay_epochs = 10
batch_size = 1024  # the number of images per forward/backward pass

now = datetime.datetime.now()
checkpoint_folder_path = f'D:\Data\CycleGAN_Model\subject_{subject}\Experiment_{version}\models\check_points'
train_model = CycleGAN_Training.ModelTraining(num_epochs, batch_size, decay_epochs, display_step=int(sample_number/batch_size)+1)
gan_models, generated_old_data = train_model.trainModel(old_LWLW_data, new_LWLW_data, checkpoint_folder_path)
print(datetime.datetime.now() - now)


##  save trained gan models
model_type = 'CycleGAN'
model_name = ['gen_AB', 'gen_BA', 'disc_A', 'disc_B']
Model_Storage.saveModels(gan_models, subject, version, model_type, model_name, project='CycleGAN_Model')


# ## generate new data
# new_LWSA_data = np.vstack(old_emg_images['emg_LWSA'])  # size [n_samples, n_channel, length, width]
# batch_size = 1024
# now = datetime.datetime.now()
# generator_model = GAN_Testing.ModelTesting(gan_models['gen_BA'], batch_size)
# fake_old_LWLW_data = generator_model.testModel(new_LWLW_data)
# fake_old_LWSA_data = generator_model.testModel(new_LWSA_data)
# print(datetime.datetime.now() - now)
#
# ## discriminate data
# discriminator_model = GAN_Testing.ModelTesting(gan_models['disc_A'], batch_size)
# discriminate_fake_LWLW = discriminator_model.testModel(fake_old_LWLW_data)
# discriminate_fake_LWSA = discriminator_model.testModel(fake_old_LWSA_data)





'''classify generated data'''
## load pretrained classification model
subject = 'Number4'
version = 0  # the data from which experiment version to process
model_type = 'Raw_Cnn2d'  # Note: it requires reordering=True when train this model, in order to match the order of gan-generated data
model_name = list(range(5))
classify_models = Model_Storage.loadModels(subject, version, model_type, model_name, project='Insole_Emg')


## generate fake data
# load trained gan models
subject = 'Test'
version = 1  # the data from which experiment version to process
model_type = 'CycleGAN'
model_name = ['gen_AB', 'gen_BA', 'disc_A', 'disc_B']
batch_size = 8192
gan_models = Model_Storage.loadModels(subject, version, model_type, model_name, project='CycleGAN_Model')

## load gan models at certain checkpoints
checkpoint_folder_path = f'D:\Data\Generative_Model\subject_{subject}\Experiment_{version}\models\check_points'
gan_models = Model_Storage.loadCheckPointModels(checkpoint_folder_path, model_name, epoch_number=300)

# fake_emg = Data_Processing.generateFakeEmg(gan_models['gen_AB'], old_emg_normalized, start_before_toeoff_ms, endtime_after_toeoff_ms, batch_size)
fake_emg = Data_Processing.generateFakeEmg(gan_models['gen_BA'], new_emg_normalized, start_before_toeoff_ms, endtime_after_toeoff_ms,
    batch_size, sample_rate=sample_rate)

## substitute some fake emg by real emg
# emg_NOT_to_substitute = ['emg_LWLW']  # the transition type to substitute
emg_NOT_to_substitute = 'all'  # using fake EMG
# emg_NOT_to_substitute = []  # using original emg
# generated_emg_data = Data_Processing.substituteFakeImages(fake_emg, old_emg_preprocessed, limit, emg_NOT_to_substitute=emg_NOT_to_substitute)
generated_emg_data = Data_Processing.substituteFakeImages(fake_emg, new_emg_preprocessed, limit, emg_NOT_to_substitute=emg_NOT_to_substitute)


## process generated data
now = datetime.datetime.now()
fold = 5  # 5-fold cross validation
cross_validation_groups = Data_Preparation.crossValidationSet(fold, generated_emg_data)
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
    Sliding_Ann_Results.saveModelResults(subject, model_results[result_set], version, result_set, window_parameters, model_type, project='Cycle_GAN')


## majority vote results without prior information, with sliding windows
overall_accuracy = []
overall_cm_recall = []
for model_result in model_results:
    sliding_majority_vote = Sliding_Ann_Results.majorityVoteResults(model_result, feature_window_per_repetition, predict_window_shift_unit,
        predict_using_window_number, initial_start=0)
    accuracy_allgroup, cm_allgroup = Data_Processing.slidingMvResults(sliding_majority_vote)
    average_accuracy_with_delay, average_cm_recall_with_delay = Data_Processing.averageAccuracyCm(accuracy_allgroup, cm_allgroup,
        feature_window_increment_ms, predict_window_shift_unit)
    overall_accuracy.append(average_accuracy_with_delay)
    overall_cm_recall.append(average_cm_recall_with_delay)
average_accuracy, average_cm_recall = Data_Processing.getAverageResults(overall_accuracy, overall_cm_recall)


## majority vote results using prior information, with sliding windows to get predict results at different delay points
overall_accuracy_by_group = []
overall_cm_recall_by_group = []
for model_result in model_results:
    reorganized_results = MV_Results_ByGroup.groupedModelResults(model_result)
    sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results, feature_window_per_repetition,
        predict_window_shift_unit, predict_using_window_number, initial_start=0)
    accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
    # calculate the accuracy and cm. Note: the first dimension refers to each delay
    average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
        cm_bygroup)
    accuracy_by_group, cm_recall_by_group = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
        predict_window_shift_unit)
    overall_accuracy_by_group.append(accuracy_by_group)
    overall_cm_recall_by_group.append(cm_recall_by_group)
average_accuracy_by_group, average_cm_recall_by_group = Data_Processing.getAverageResults(overall_accuracy_by_group, overall_cm_recall_by_group)



