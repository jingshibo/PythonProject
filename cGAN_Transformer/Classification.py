##
from Conditional_GAN.Data_Procesing import Process_Raw_Data, Train_Classifiers, Train_cGan
from cGAN_Transformer.Functions import Preprocessing, Results, Storage, Plot_Raw_Data
from cGAN_Transformer.Models import Classification_Model, Transformer_GAN_Training, Transformer_GAN_Testing
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix
import datetime
import copy


## load and filter data
subject = 'Number1'
grid = 'grid_1'
version = 0  # the data from which experiment version to process
up_down_session_t0 = [0, 1, 2, 3, 4]
down_up_session_t0 = [1, 2, 3, 4, 5]
up_down_session_t1 = [0, 1, 2, 3, 4]
down_up_session_t1 = [1, 2, 3, 4, 5]
old_emg_data, new_emg_data, window_parameters, start_before_toeoff_ms = Train_cGan.realEmgData(subject, version, up_down_session_t0,
    down_up_session_t0, up_down_session_t1, down_up_session_t1, grid=grid, envelope=True, envelope_cutoff=50, reordering=False)


## parameters for normalize emg data to train models
amplitude_limit = 1500
spatial_filter_kernel = (2, 1)
modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'],
    'emg_SALW': ['emg_SASA', 'emg_LWLW', 'emg_SALW'],
    'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}  # The order in each list is important, corresponding to gen_data_1 and gen_data_2.
transition_encoding = {"emg_LWSA": 0, "emg_LWSD": 1, "emg_SALW": 2, "emg_SDLW": 3}  # encode the conditions into integer
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']  # the length of data in each repetition
old_emg_normalized, new_emg_normalized, _, _ = Process_Raw_Data.normalizeFilterEmgData(old_emg_data, new_emg_data, amplitude_limit,
    normalize='(0,1)', spatial_filter=False, kernel=spatial_filter_kernel)
old_emg_aligned, old_lags_butter = Preprocessing.align_by_cross_correlation(old_emg_normalized, max_lag=100, num_iterations=3,
    initial_reference_method='average', verbose=False, butter_cutoff_freq=20, butter_filter_order=4)
new_emg_aligned, new_lags_butter = Preprocessing.align_by_cross_correlation(new_emg_normalized, max_lag=100, num_iterations=3,
    initial_reference_method='average', verbose=False, butter_cutoff_freq=20, butter_filter_order=4)


## parameters for extracting emg data to train models
window_length = 1200
window_increment = 100
num_window_per_transition = 1
window_shift = 0
channel_shift = 15
start = - window_length - window_shift
end = window_increment * num_window_per_transition + window_shift
time_range = [-600, 600]  # select data centered around toe-off
old_emg_central, new_emg_central, train_gan_data = Preprocessing.extractGanTrainingData(modes_generation, old_emg_aligned,
    new_emg_aligned, time_range)
classify_old_emg = Preprocessing.buildClassifyDataset(old_emg_central)


## plot raw data
transition_types = ['emg_LWSA', 'emg_LWSD', 'emg_SALW', 'emg_SDLW']
transition_type = 'emg_SASA'
time_point = 0
time_slice_start = window_shift + time_point * window_increment
time_slice_end = time_slice_start + window_length
# Plot_Raw_Data.plot_overlap_sample_all_modes(old_emg_central)
# Plot_Raw_Data.plot_time_series_and_heatmaps(old_emg_central[transition_type], time_start=time_slice_start, time_end=time_slice_end,
# key_label=transition_type, num_samples=5, y_limit=(0, 0.4))
# Plot_Raw_Data.plot_heatmaps_samples(old_emg_central[transition_type], time_start=0, time_end=None,
#     key_label=transition_type, num_samples=5, y_limit=(0, 0.4), random_sampling=True, stata='mean')
# Plot_Raw_Data.plot_time_series_samples(old_emg_central[transition_type], time_start=0, time_end=None,
#     key_label=transition_type, num_samples=30, y_limit=(0, 0.4), random_sampling=False, stata='mean')


## Stack and reshape
# import numpy as np
# from scipy.signal import butter, sosfiltfilt
# data_list = old_emg_central['emg_SDLW']  # List of 60 arrays, each of shape (1200, 65)
# data_4d = np.stack([sample.T for sample in data_list], axis=0)  # Now shape (60, 65, 1200)
# data_4d = data_4d[:, np.newaxis, :, :]  # Add channel dim → shape (60, 1, 65, 1200)
#
# sos = butter(4, 10, fs=1000, btype='lowpass', output='sos')
# data = data_4d[:, 0, :, :]  # Reshape: (60, 1, 65, 1200) → (60, 65, 1200)
# filtered = sosfiltfilt(sos, data, axis=-1)  # Apply along axis=-1 (time)
#
# Plot_Raw_Data.plot_time_series_samples(np.transpose(filtered, (0, 2, 1)), time_start=0, time_end=None,
#     key_label=transition_type, num_samples=5, y_limit=(0, 0.4), random_sampling=True)


## train gan model
NUM_EPOCHS = 30
num_batch_per_epoch = 50
num_sample_per_condition = 5  # in each batch
num_transitions = len(transition_encoding)
BATCH_SIZE = num_sample_per_condition * num_transitions * num_window_per_transition
model_type = 'Transformer_Gan'
model_name = ['gen', 'disc']
training_parameters = {'modes_generation': modes_generation, 'num_epochs': NUM_EPOCHS, 'num_batch_per_epoch': num_batch_per_epoch,
    'num_sample_per_condition': num_sample_per_condition, 'window_length': window_length, 'window_increment': window_increment,
    'num_window_per_transition': num_window_per_transition, 'window_shift': window_shift, 'channel_shift': channel_shift}
storage_parameters = {'subject': subject, 'version': version, 'model_type': model_type, 'model_name': model_name, 'gan_result_set': 0}
trainer = Transformer_GAN_Training.GanTraining(NUM_EPOCHS, num_sample_per_condition, num_batch_per_epoch)
model = trainer.trainModel(train_gan_data, transition_encoding, training_parameters, storage_parameters)



## generate transition data
epoch_number = 30
model = Storage.loadCheckPointModels(storage_parameters, epoch_number)
generated_transition_data = Transformer_GAN_Testing.generateTransitionData(model['gen'], train_gan_data, transition_encoding,
    num_window_per_transition, window_length, window_increment, window_shift, sample_number=30, batch_size=5)



## print image
transition_types = ['emg_LWSA', 'emg_LWSD', 'emg_SALW', 'emg_SDLW']
transition_type = 'emg_LWSA'
time_point = 0
generated_image = generated_transition_data[transition_type][time_point]['generated_images'].squeeze(1)
blending_factor_A = generated_transition_data[transition_type][time_point]['blending_factors'][:, 0, :, :]
blending_factor_B = generated_transition_data[transition_type][time_point]['blending_factors'][:, 1, :, :]
# Plot_Raw_Data.plot_heatmaps_samples(generated_image, key_label=transition_type, num_samples=5, y_limit=(0, 0.4), stata='mean')
Plot_Raw_Data.plot_time_series_samples(generated_image, key_label=transition_type, num_samples=5, y_limit=(0, 0.4), stata='mean')
Plot_Raw_Data.plot_time_series_samples(blending_factor_A, key_label=transition_type, num_samples=5, y_limit=(0, 1))
Plot_Raw_Data.plot_time_series_samples(blending_factor_B, key_label=transition_type, num_samples=5, y_limit=(0, 1))


# Plot_Raw_Data.plot_heatmaps_samples(old_emg_central[transition_type], time_start=0, time_end=None,
#     key_label=transition_type, num_samples=5, y_limit=(0, 0.4), random_sampling=True, stata='mean')
Plot_Raw_Data.plot_time_series_samples(old_emg_central[transition_type], time_start=0, time_end=None,
    key_label=transition_type, num_samples=5, y_limit=(0, 0.4), random_sampling=True, stata='mean')



##  classify using a single cnn 2d model
# num_epochs = 50
# batch_size = 32
# decay_epochs = 20
# now = datetime.datetime.now()
# fold_number = 5
# cross_validation_indices, cross_validation_dataset = Preprocessing.crossValidationSet(fold_number, classify_old_emg)
# train_model = Classification_Model.ModelTraining(num_epochs, batch_size, report_period=10)
# models, model_results = train_model.trainModel(classify_old_emg, cross_validation_indices, decay_epochs)
# print(datetime.datetime.now() - now)
# accuracy, cm_recall = Results.getAccuracyCm(model_results)
# class_labels = ['LW', 'LWSA', 'LWSD', 'SALW', 'SA', 'SDLW', 'SD']
# Confusion_Matrix.plotConfusionMatrix(cm_recall, class_labels, normalize=False)

# import winsound
# winsound.Beep(frequency=1000, duration=1000)