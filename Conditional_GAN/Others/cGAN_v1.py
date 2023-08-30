##

from Transition_Prediction.Pre_Processing import Preprocessing
from Transition_Prediction.Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Transition_Prediction.Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Cycle_GAN.Functions import Data_Processing, Classify_Testing
from Conditional_GAN.Models import cGAN_Training, cGAN_Testing, Model_Storage
from Conditional_GAN.Data_Procesing import Process_Fake_Data
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset, Raw_Cnn2d_Model
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
predict_window_per_repetition = int(
    (endtime_after_toeoff_ms + start_before_toeoff_ms - predict_window_ms) / predict_window_increment_ms) + 1


## old data
subject = 'Number4'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
# up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# down_up_session = [0, 1, 2, 5, 6, 7, 8, 9, 10]
up_down_session = [0, 1, 2, 3, 4]
down_up_session = [0, 1, 2, 5, 6]
# up_down_session = [0]
# down_up_session = [0]
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
# up_down_session = [0]
# down_up_session = [0]
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
old_emg_normalized = Data_Processing.normalizeEmgData(old_emg_preprocessed, range_limit=limit)
new_emg_normalized = Data_Processing.normalizeEmgData(new_emg_preprocessed, range_limit=limit)
old_emg_reshaped = {k: [np.transpose(np.reshape(arr, newshape=(-1, 13, 10, 1), order='F'), (0, 3, 1, 2)).astype(np.float32) for arr in v]
    for k, v in old_emg_normalized.items()}
# del old_emg_preprocessed
new_emg_reshaped = {k: [np.transpose(np.reshape(arr, newshape=(-1, 13, 10, 1), order='F'), (0, 3, 1, 2)).astype(np.float32) for arr in v]
    for k, v in new_emg_normalized.items()}
# del new_emg_preprocessed


## model data
subject = 'Test'
version = 1  # the data from which experiment version to process
old_LWLW_data = np.vstack(old_emg_reshaped['emg_LWLW'])
old_SASA_data = np.vstack(old_emg_reshaped['emg_SASA'])
old_LWSA_data = np.vstack(old_emg_reshaped['emg_LWSA'])
new_LWLW_data = np.vstack(new_emg_reshaped['emg_LWLW'])
new_SASA_data = np.vstack(new_emg_reshaped['emg_SASA'])
new_LWSA_data = np.vstack(new_emg_reshaped['emg_LWSA'])


## seperate dataset by timepoints
time_interval = 50
period = start_before_toeoff_ms + endtime_after_toeoff_ms
separated_old_LWLW = Process_Fake_Data.separateByTimeInterval(old_LWLW_data, timepoint_interval=time_interval, length=period, output_list=True)
separated_old_SASA = Process_Fake_Data.separateByTimeInterval(old_SASA_data, timepoint_interval=time_interval, length=period, output_list=True)
separated_old_LWSA = Process_Fake_Data.separateByTimeInterval(old_LWSA_data, timepoint_interval=time_interval, length=period, output_list=True)
train_data = {'gen_data_1': separated_old_LWLW, 'gen_data_2': separated_old_SASA, 'disc_data': separated_old_LWSA}




'''
    train generative models
'''
## storage information
checkpoint_model_path = f'D:\Data\cGAN_Model\subject_{subject}\Experiment_{version}\models\check_points'
checkpoint_result_path = f'D:\Data\cGAN_Model\subject_{subject}\Experiment_{version}\model_results\check_points'
model_type = 'cGAN'
model_name = ['gen', 'disc']
result_set = 0


## hyperparameters
num_epochs = 400  # the number of times you iterate through the entire dataset when training
decay_epochs = 10
batch_size = 1024  # the number of images per forward/backward pass
sampling_repetition = 6  # the number of batches to repeat the combination sampling for the same time points
noise_dim = 0
blending_factor_dim = 2

now = datetime.datetime.now()
train_model = cGAN_Training.ModelTraining(num_epochs, batch_size, sampling_repetition, decay_epochs, noise_dim, blending_factor_dim)
gan_models, blending_factors = train_model.trainModel(train_data, checkpoint_model_path, checkpoint_result_path)
print(datetime.datetime.now() - now)


## save trained gan models
Model_Storage.saveModels(gan_models, subject, version, model_type, model_name, project='cGAN_Model')


## save model results
training_parameters = {'start_before_toeoff_ms': start_before_toeoff_ms, 'endtime_after_toeoff_ms': endtime_after_toeoff_ms,
    'noise_dim': noise_dim, 'sampling_repetition': sampling_repetition, 'batch_size': batch_size, 'num_epochs': num_epochs,
    'interval': time_interval}
Model_Storage.saveCGanResults(subject, blending_factors, version, result_set, training_parameters, model_type, project='cGAN_Model')



'''
    build classifier (on subject 4)
'''
## load blending factors
gen_results = Model_Storage.loadCGanResults(subject, version, result_set, model_type, project='cGAN_Model')
estimated_blending_factors = {key: np.array(value) for key, value in gen_results['model_results'].items()}


## separate dataset into 1ms interval
period = start_before_toeoff_ms + endtime_after_toeoff_ms
reorganized_old_LWLW = Process_Fake_Data.separateByTimeInterval(old_LWLW_data, timepoint_interval=1, length=period)
reorganized_old_SASA = Process_Fake_Data.separateByTimeInterval(old_SASA_data, timepoint_interval=1, length=period)
reorganized_data = {'gen_data_1': reorganized_old_LWLW, 'gen_data_2': reorganized_old_SASA, 'blending_factors': estimated_blending_factors}


## generate fake data at each timestamp
interval = gen_results['training_parameters']['interval']
fake_data = Process_Fake_Data.generateFakeData(reorganized_data, interval, repetition=1, random_pairing=False)
reorganized_fake_data = Process_Fake_Data.reorganizeFakeData(fake_data)
# build training data
fake_emg_data = {'emg_LWSA': reorganized_fake_data}
generated_data = Process_Fake_Data.replaceUsingFakeEmg(fake_emg_data, old_emg_normalized)


## build training set
leave_percentage = 0.2
leave_one_groups = Data_Preparation.leaveOutDataSet(leave_percentage, generated_data, shuffle=True)
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(leave_one_groups, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups


##  train the classify using fake data
num_epochs = 50
batch_size = 1024
decay_epochs = 20
now = datetime.datetime.now()
# train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size)
train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size, report_period=10)
models, model_results = train_model.trainModel(shuffled_groups, decay_epochs)  # only output one model instead of five models
print(datetime.datetime.now() - now)


## test fake data performance
reorganized_results = MV_Results_ByGroup.groupedModelResults(model_results)
sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results, feature_window_per_repetition,
    predict_window_shift_unit, predict_using_window_number, initial_start=0)
accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
# calculate the accuracy and cm. Note: the first dimension refers to each delay
average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
    cm_bygroup)
accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
    predict_window_shift_unit)


## build real data test set
test_ratio = 0.5
real_test_data = Process_Fake_Data.replaceUsingRealEmg(list(fake_emg_data.keys()), old_emg_normalized, leave_one_groups, test_ratio)
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(real_test_data, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups


## classify real data
batch_size = 1024
now = datetime.datetime.now()
test_model = Classify_Testing.ModelTesting(models[0], batch_size)  # select one pretrained model for classifying data
test_result = test_model.testModel(shuffled_groups)
print(datetime.datetime.now() - now)


## test real data performance
reorganized_results = MV_Results_ByGroup.groupedModelResults(test_result)
sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results, feature_window_per_repetition,
    predict_window_shift_unit, predict_using_window_number, initial_start=0)
accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
# calculate the accuracy and cm. Note: the first dimension refers to each delay
average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
    cm_bygroup)
accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
    predict_window_shift_unit)



'''
    training classifier (on subject 5)
'''
## separate dataset into 1ms interval
period = start_before_toeoff_ms + endtime_after_toeoff_ms
reorganized_new_LWLW = Process_Fake_Data.separateByTimeInterval(new_LWLW_data, timepoint_interval=1, length=period)
reorganized_new_SASA = Process_Fake_Data.separateByTimeInterval(new_SASA_data, timepoint_interval=1, length=period)
reorganized_data = {'gen_data_1': reorganized_new_LWLW, 'gen_data_2': reorganized_new_SASA, 'blending_factors': estimated_blending_factors}


## generate fake data at each timestamp
interval = gen_results['training_parameters']['interval']
fake_data = Process_Fake_Data.generateFakeData(reorganized_data, interval, repetition=1, random_pairing=False)
reorganized_fake_data = Process_Fake_Data.reorganizeFakeData(fake_data)
# build training data
fake_emg_data = {'emg_LWSA': reorganized_fake_data}
generated_data = Process_Fake_Data.replaceUsingFakeEmg(fake_emg_data, new_emg_normalized)


## build training set
leave_percentage = 0.2
leave_one_groups = Data_Preparation.leaveOutDataSet(leave_percentage, generated_data, shuffle=True)
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(leave_one_groups, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups


##  train the classify using fake data
num_epochs = 50
batch_size = 1024
decay_epochs = 20
now = datetime.datetime.now()
# train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size)
train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size, report_period=10)
models, model_results = train_model.trainModel(shuffled_groups, decay_epochs)  # only output one model instead of five models
print(datetime.datetime.now() - now)


## test fake data performance
reorganized_results = MV_Results_ByGroup.groupedModelResults(model_results)
sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results, feature_window_per_repetition,
    predict_window_shift_unit, predict_using_window_number, initial_start=0)
accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
# calculate the accuracy and cm. Note: the first dimension refers to each delay
average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
    cm_bygroup)
accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
    predict_window_shift_unit)


## build real data test set
test_ratio = 0.5
real_test_data = Process_Fake_Data.replaceUsingRealEmg(list(fake_emg_data.keys()), new_emg_normalized, leave_one_groups, test_ratio)
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(real_test_data, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups


## classify real data
batch_size = 1024
now = datetime.datetime.now()
test_model = Classify_Testing.ModelTesting(models[0], batch_size)  # select one pretrained model for classifying data
test_result = test_model.testModel(shuffled_groups)
print(datetime.datetime.now() - now)


## test real data performance
reorganized_results = MV_Results_ByGroup.groupedModelResults(test_result)
sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results, feature_window_per_repetition,
    predict_window_shift_unit, predict_using_window_number, initial_start=0)
accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
# calculate the accuracy and cm. Note: the first dimension refers to each delay
average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
    cm_bygroup)
accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
    predict_window_shift_unit)





'''
    Comparison
'''
##
# build training data
old_emg_data = {'emg_LWSA': old_emg_normalized['emg_LWSA']}
generated_data = Process_Fake_Data.replaceUsingFakeEmg(old_emg_data, new_emg_normalized)

## build training set
leave_percentage = 0.8
leave_one_groups = Data_Preparation.leaveOutDataSet(leave_percentage, generated_data, shuffle=True)
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(leave_one_groups, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups


##  train the classify using fake data
num_epochs = 50
batch_size = 1024
decay_epochs = 20
now = datetime.datetime.now()
# train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size)
train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size, report_period=10)
models, model_results = train_model.trainModel(shuffled_groups, decay_epochs)  # only output one model instead of five models
print(datetime.datetime.now() - now)


## test fake data performance
reorganized_results = MV_Results_ByGroup.groupedModelResults(model_results)
sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results, feature_window_per_repetition,
    predict_window_shift_unit, predict_using_window_number, initial_start=0)
accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
# calculate the accuracy and cm. Note: the first dimension refers to each delay
average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
    cm_bygroup)
accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
    predict_window_shift_unit)


## build real data test set
test_ratio = 0.5
real_test_data = Process_Fake_Data.replaceUsingRealEmg('emg_LWSA', new_emg_normalized, leave_one_groups, test_ratio)
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(real_test_data, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups


## classify real data
batch_size = 1024
now = datetime.datetime.now()
test_model = Classify_Testing.ModelTesting(models[0], batch_size)  # select one pretrained model for classifying data
test_result = test_model.testModel(shuffled_groups)
print(datetime.datetime.now() - now)


## test real data performance
reorganized_results = MV_Results_ByGroup.groupedModelResults(test_result)
sliding_majority_vote_by_group = Sliding_Ann_Results.SlidingMvResultsByGroup(reorganized_results, feature_window_per_repetition,
    predict_window_shift_unit, predict_using_window_number, initial_start=0)
accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
# calculate the accuracy and cm. Note: the first dimension refers to each delay
average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
    cm_bygroup)
accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
    predict_window_shift_unit)





## load check point results
checkpoint_result = Model_Storage.loadCheckPointCGanResults(checkpoint_result_path, 400)


## load check point models
test_model = Model_Storage.loadCheckPointModels(checkpoint_model_path, model_name, epoch_number=50)
gen_model = cGAN_Testing.ModelTesting(test_model['gen'])
results = gen_model.estimateBlendingFactors(train_data, noise_dim=0)



