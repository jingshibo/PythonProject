##
from Pre_Processing import Preprocessing
from Models.Utility_Functions import Data_Preparation
import datetime
import numpy as np
from Generative_Model.Functions import CycleGAN_Training, GAN_Testing, Model_Storage, Visualization, Data_Processing


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
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
del emg_filtered_data
old_emg_images = {k: [np.transpose(np.reshape(arr, newshape=(-1, 13, 10, 1), order='F'), (0, 3, 1, 2)).astype(np.float32) for arr in v] for
    k, v in emg_preprocessed.items()}
del emg_preprocessed


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
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
del emg_filtered_data
new_emg_images = {k: [np.transpose(np.reshape(arr, newshape=(-1, 13, 10, 1), order='F'), (0, 3, 1, 2)).astype(np.float32) for arr in v] for
    k, v in emg_preprocessed.items()}
del emg_preprocessed


## visualize data distribution
Visualization.plotHistPercentage(old_emg_images)
Visualization.plotHistByClass(old_emg_images)


## clip the values and normalize data
limit = 1500
old_emg_normalized = Data_Processing.normalizeEmgData(old_emg_images, limit=limit)
new_emg_normalized = Data_Processing.normalizeEmgData(new_emg_images, limit=limit)


##  train generative models
old_LWLW_data = old_emg_normalized['emg_LWLW']
new_LWLW_data = new_emg_normalized['emg_LWLW']

num_epochs = 10  # the number of times you iterate through the entire dataset when training
decay_epochs = 4
batch_size = 1024  # the number of images per forward/backward pass

now = datetime.datetime.now()
train_model = CycleGAN_Training.ModelTraining(num_epochs, batch_size, decay_epochs, display_step=int(len(old_LWLW_data)/batch_size))
gan_models = train_model.trainModel(old_LWLW_data, new_LWLW_data)
print(datetime.datetime.now() - now)


##  save gan models
# model path
subject = 'Test'
version = 1  # the data from which experiment version to process
model_type = 'CycleGAN'
model_name = ['gen_AB', 'gen_BA', 'disc_A', 'disc_B']
Model_Storage.saveModels(gan_models, subject, version, model_type, model_name, project='Generative_Model')


##  load gan models
gan_models = Model_Storage.loadModels(subject, version, model_type, model_name, project='Generative_Model')


## generate new data
new_LWSA_data = np.vstack(old_emg_images['emg_LWSA'])  # size [n_samples, n_channel, length, width]
batch_size = 1024
now = datetime.datetime.now()
generator_model = GAN_Testing.ModelTesting(gan_models['gen_BA'], batch_size)
fake_old_LWSA_data = generator_model.testModel(new_LWLW_data)
print(datetime.datetime.now() - now)


## discriminate data
discriminator_model = GAN_Testing.ModelTesting(gan_models['disc_A'], batch_size)
discriminate_fake_old = discriminator_model.testModel(fake_old_LWSA_data)

