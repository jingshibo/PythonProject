##
from Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Cycle_GAN.Functions import Data_Processing
from Conditional_GAN.Functions import PreProcessing, cGAN_Training
from Cycle_GAN.Functions import Model_Storage
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


## model data
subject = 'Test'
version = 1  # the data from which experiment version to process
old_LWLW_data = old_emg_normalized['emg_LWLW']
old_SASA_data = old_emg_normalized['emg_SASA']
old_LWSA_data = old_emg_normalized['emg_LWSA']


## seperate dataset by timepoints
timepoint_interval = 50
separated_old_LWLW = PreProcessing.separateByTimepoints(old_LWLW_data, timepoint_interval=timepoint_interval,
    period=start_before_toeoff_ms + endtime_after_toeoff_ms)
separated_old_SASA = PreProcessing.separateByTimepoints(old_SASA_data, timepoint_interval=timepoint_interval,
    period=start_before_toeoff_ms + endtime_after_toeoff_ms)
separated_old_LWSA = PreProcessing.separateByTimepoints(old_LWSA_data, timepoint_interval=timepoint_interval,
    period=start_before_toeoff_ms + endtime_after_toeoff_ms)
train_data = {'gen_data_1': separated_old_LWLW, 'gen_data_2': separated_old_SASA, 'disc_data': separated_old_LWSA,}


## train generative models
# hyperparameters
num_epochs = 400  # the number of times you iterate through the entire dataset when training
decay_epochs = 10
batch_size = 1024  # the number of images per forward/backward pass
sampling_repetition = 4  # the number of batches to repeat the combination sampling for the same time points
noise_dim = 0

now = datetime.datetime.now()
checkpoint_folder_path = f'D:\Data\cGAN_Model\subject_{subject}\Experiment_{version}\models\check_points'
train_model = cGAN_Training.ModelTraining(num_epochs, batch_size, sampling_repetition, decay_epochs, noise_dim)
gan_models, blending_factors = train_model.trainModel(train_data, checkpoint_folder_path)
print(datetime.datetime.now() - now)


## save trained gan models
model_type = 'cGAN'
model_name = ['gen', 'disc']
Model_Storage.saveModels(gan_models, subject, version, model_type, model_name, project='cGAN')


## save model results
model_type = 'cGAN'
result_set = 0
model_parameters = {'start_before_toeoff_ms': start_before_toeoff_ms, 'endtime_after_toeoff_ms': endtime_after_toeoff_ms,
    'noise_dim': noise_dim, 'sampling_repetition':sampling_repetition, 'batch_size': batch_size, 'num_epochs':num_epochs}
Model_Storage.saveGanResults(subject, blending_factors, version, result_set, model_parameters, model_type, project='cGAN_Model')


## read model results



## train classification model
