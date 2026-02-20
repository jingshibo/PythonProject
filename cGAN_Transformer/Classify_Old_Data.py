##
from Conditional_GAN.Data_Procesing import Process_Raw_Data, Train_Classifiers, Train_cGan
from cGAN_Transformer.Functions import Preprocessing, Result_Analysis, Storage, Plot_Raw_Data
from cGAN_Transformer.Models import Classification_Model, Transformer_GAN_Training, Transformer_GAN_Testing, Transformer_GAN_Model
from Transition_Prediction.Models.Utility_Functions import Confusion_Matrix
from Conditional_GAN.Data_Procesing import Dtw_Similarity
import numpy as np
import datetime
import copy


## load and filter data
subject = 'Number1'
grid = 'grid_1'
version = 0  # the data from which experiment version to process
result_set = 0
up_down_session_t0 = [0, 1, 2, 3, 4]
down_up_session_t0 = [1, 2, 3, 4, 5]
up_down_session_t1 = [0, 1, 2, 3, 4]
down_up_session_t1 = [1, 2, 3, 4]
old_emg_data, new_emg_data, window_parameters, start_before_toeoff_ms = Train_cGan.realEmgData(subject, version, up_down_session_t0,
    down_up_session_t0, up_down_session_t1, down_up_session_t1, grid=grid, envelope=True, envelope_cutoff=400, reordering=False)


## parameters for normalize emg data to train models
amplitude_limit = 1500
class_labels = ['LW', 'LWSA', 'LWSD', 'SALW', 'SA', 'SDLW', 'SD']
modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'],
    'emg_SALW': ['emg_SASA', 'emg_LWLW', 'emg_SALW'],
    'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}  # The order in each list is important, corresponding to gen_data_1 and gen_data_2.
transition_encoding = {"emg_LWSA": 0, "emg_LWSD": 1, "emg_SALW": 2, "emg_SDLW": 3}  # encode the conditions into integer
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']  # the length of data in each repetition
old_emg_normalized, new_emg_normalized, _, _ = Process_Raw_Data.normalizeFilterEmgData(old_emg_data, new_emg_data, amplitude_limit,
    normalize='(0,1)', spatial_filter=False, kernel=(2, 1))
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
old_emg_central, new_emg_central, train_gan_data, new_gan_data = Preprocessing.extractGanTrainingData(modes_generation, old_emg_aligned,
    new_emg_aligned, time_range)


## plot raw data
transition_types = ['emg_LWSA', 'emg_LWSD', 'emg_SALW', 'emg_SDLW', 'emg_LWLW', 'emg_SASA', 'emg_SDSD']
transition_type = 'emg_LWSA'
time_point = 0
time_slice_start = window_shift + time_point * window_increment
time_slice_end = time_slice_start + window_length
# Plot_Raw_Data.plot_overlap_sample_all_modes(old_emg_central, y_limit=(0, 1))
# Plot_Raw_Data.plot_time_series_and_heatmaps(old_emg_central[transition_type], time_start=time_slice_start, time_end=time_slice_end,
# key_label=transition_type, num_samples=5, y_limit=(0, 0.4))
# Plot_Raw_Data.plot_heatmaps_samples(old_emg_central[transition_type], time_start=0, time_end=None,
#     key_label=transition_type, num_samples=60, y_limit=(0, 0.4), random_sampling=True, stata='mean')
# Plot_Raw_Data.plot_time_series_samples(old_emg_central[transition_type], time_start=0, time_end=None,
#     key_label=transition_type, num_samples=60, y_limit=(0, 1), random_sampling=False, stata='mean')


## train gan model
NUM_EPOCH_TO_TRAIN = 100
num_batch_per_epoch = 50
num_sample_per_condition = 5  # in each batch
num_transitions = len(transition_encoding)
BATCH_SIZE = num_sample_per_condition * num_transitions * num_window_per_transition
model_type = 'Transformer_Gan'
model_name = ['gen', 'disc']
training_parameters = {'modes_generation': modes_generation, 'num_epochs': NUM_EPOCH_TO_TRAIN, 'num_batch_per_epoch': num_batch_per_epoch,
    'num_sample_per_condition': num_sample_per_condition, 'window_length': window_length, 'window_increment': window_increment,
    'num_window_per_transition': num_window_per_transition, 'window_shift': window_shift, 'channel_shift': channel_shift}
storage_parameters = {'subject': subject, 'version': version, 'model_type': model_type, 'model_name': model_name, 'gan_result_set': 0}
gen_model = 'two_factors'  # one_factor or two_factors
trainer = Transformer_GAN_Training.GanTraining(NUM_EPOCH_TO_TRAIN, num_sample_per_condition, num_batch_per_epoch)
gan_models = trainer.trainModel(train_gan_data, transition_encoding, training_parameters, storage_parameters, gen_model)


## generate transition data
num_conditions = len(transition_encoding) * training_parameters['num_window_per_transition']
model_class_map = {'gen': Transformer_GAN_Model.EMGFusionTwoFactorGenerator(num_conditions),
    'disc': Transformer_GAN_Model.EMGFusionPatchDiscriminator(num_conditions)}
epoch_number_to_generate = 100
gan_model = Storage.loadCheckPointModels(storage_parameters, epoch_number_to_generate, model_class_map, gen_model)
all_generated_data = Transformer_GAN_Testing.generateTransitionData(gan_model['gen'], train_gan_data, transition_encoding,
    num_window_per_transition, window_length, window_increment, window_shift, number_to_generate=3600, batch_size=30)
time_0_fake_data, ordered_sampled_data = Transformer_GAN_Testing.returnDataForPlotting(all_generated_data, ordered_sample_number=50)


## sampled results
extracted_data = Dtw_Similarity.extractFakeData(time_0_fake_data, old_emg_central, modes_generation, envelope_frequency=30, num_sample=50,
    num_reference=30, method='random', random_reference=False, split_grids=True)
selected_fake_data, organized_fake_data = Transformer_GAN_Testing.fakeDataForTraining(extracted_data)


## print image
transition_types = ['emg_LWSA', 'emg_LWSD', 'emg_SALW', 'emg_SDLW']
transition_type = 'emg_SALW'
time_point = 0
generated_image = selected_fake_data[transition_type][time_point]['generated_images'].squeeze(1)
blending_factor_A = ordered_sampled_data[transition_type][time_point]['blending_factors'][:, 0, :, :]
# blending_factor_B = ordered_sampled_data[transition_type][time_point]['blending_factors'][:, 1, :, :]

Plot_Raw_Data.plot_heatmaps_samples(generated_image, key_label=transition_type, num_samples=30, y_limit=(0, 0.5), stata='mean')
Plot_Raw_Data.plot_time_series_samples(generated_image, key_label=transition_type, num_samples=30, y_limit=(0, 0.5), stata='mean')
# Plot_Raw_Data.plot_time_series_samples(blending_factor_A, key_label=transition_type, num_samples=5, y_limit=(0, 1))
# Plot_Raw_Data.plot_time_series_samples(blending_factor_B, key_label=transition_type, num_samples=5, y_limit=(0, 1))
# Plot_Raw_Data.plot_sample_fft(generated_image, transition_type, num_samples=5)


##
transition_types = ['emg_LWSA', 'emg_LWSD', 'emg_SALW', 'emg_SDLW', 'emg_LWLW', 'emg_SASA', 'emg_SDSD']
transition_type = 'emg_SALW'
Plot_Raw_Data.plot_heatmaps_samples(old_emg_central[transition_type], time_start=0, time_end=None,
    key_label=transition_type, num_samples=30, y_limit=(0, 0.5), random_sampling=False, stata='mean')
Plot_Raw_Data.plot_time_series_samples(old_emg_central[transition_type], time_start=0, time_end=None,
    key_label=transition_type, num_samples=30, y_limit=(0, 0.5), random_sampling=False, stata='mean')
# Plot_Raw_Data.plot_sample_fft(old_emg_central[transition_type], transition_type, num_samples=5)
# transition_type = 'emg_SDSD'
# Plot_Raw_Data.plot_time_series_samples(old_emg_central[transition_type], time_start=0, time_end=None,
#     key_label=transition_type, num_samples=5, y_limit=(0, 0.4), random_sampling=True, stata='mean')


##  train the old model using synthetic transition d ata
synthetic_dataset, _ = Preprocessing.build_cv_dataset_with_augmented_data(old_emg_central, organized_fake_data,
    modes_generation, n_splits=5, n_real_steady_state=50, n_synthetic_transition=50, n_real_transition=0, random_sampling=False)
train_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
models_synthetic, results_synthetic = train_model.trainModel(synthetic_dataset, decay_epochs=20)
accuracy_synthetic, cm_recall_synthetic = Result_Analysis.getAccuracyCm(results_synthetic)
Confusion_Matrix.plotConfusionMatrix(cm_recall_synthetic, class_labels, normalize=False)
# Storage.saveClassifyResult(subject, accuracy_synthetic, cm_recall_synthetic, version, result_set, 'classify_old_synthetic', gen_model, num_reference=None)


##  train the old model using real transition data
_, original_dataset = Preprocessing.build_cv_dataset_with_augmented_data(old_emg_central, organized_fake_data,
    modes_generation, n_splits=5, n_real_steady_state=50, n_synthetic_transition=0, n_real_transition=50, random_sampling=False)
train_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
models_real, results_real = train_model.trainModel(original_dataset, decay_epochs=20)
accuracy_real, cm_recall_real = Result_Analysis.getAccuracyCm(results_real)
Confusion_Matrix.plotConfusionMatrix(cm_recall_real, class_labels, normalize=False)
# Storage.saveClassifyResult(subject, accuracy_real, cm_recall_real, version, result_set, 'classify_old_real', gen_model, num_reference=None)


## train the old model with only five real transition data
synthetic_dataset, _ = Preprocessing.build_cv_dataset_with_augmented_data(old_emg_central, organized_fake_data,
    modes_generation, n_splits=5, n_real_steady_state=50, n_synthetic_transition=0, n_real_transition=5, random_sampling=False)
train_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
models_imbalance, results_imbalance = train_model.trainModel(synthetic_dataset, decay_epochs=20)
accuracy_imbalance, cm_recall_imbalance = Result_Analysis.getAccuracyCm(results_imbalance)
# Confusion_Matrix.plotConfusionMatrix(cm_recall_imbalance, class_labels, normalize=False)
Storage.saveClassifyResult(subject, accuracy_imbalance, cm_recall_imbalance, version, result_set, 'classify_old_imbalance', gen_model, num_reference=5)


## train the old model with synthetic transition data and five real transition data
synthetic_dataset, _ = Preprocessing.build_cv_dataset_with_augmented_data(old_emg_central, organized_fake_data,
    modes_generation, n_splits=5, n_real_steady_state=50, n_synthetic_transition=50, n_real_transition=5, random_sampling=False)
train_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
models_rebalanced, results_rebalanced = train_model.trainModel(synthetic_dataset, decay_epochs=20)
accuracy_rebalanced, cm_recall_rebalanced = Result_Analysis.getAccuracyCm(results_rebalanced)
# Confusion_Matrix.plotConfusionMatrix(cm_recall_rebalanced, class_labels, normalize=False)
Storage.saveClassifyResult(subject, accuracy_rebalanced, cm_recall_rebalanced, version, result_set, 'classify_old_rebalanced', gen_model, num_reference=5)


## train the old model with real data and noisy data
synthetic_dataset, _ = Preprocessing.build_cv_dataset_with_noisy_data(old_emg_central, modes_generation, snr=25, n_splits=5,
    n_real_steady_state=50, n_synthetic_transition=50, n_real_transition=5, random_sampling=False)
train_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
models_noise, results_noise = train_model.trainModel(synthetic_dataset, decay_epochs=20)
accuracy_noise, cm_recall_noise = Result_Analysis.getAccuracyCm(results_noise)
# Confusion_Matrix.plotConfusionMatrix(cm_recall_noise, class_labels, normalize=False)
Storage.saveClassifyResult(subject, accuracy_noise, cm_recall_noise, version, result_set, 'classify_old_noisy', gen_model, num_reference=5)


## train the old model with real data and copy data
synthetic_dataset, _ = Preprocessing.build_cv_dataset_with_noisy_data(old_emg_central, modes_generation, snr=None, n_splits=5,
    n_real_steady_state=50, n_synthetic_transition=50, n_real_transition=5, random_sampling=False)
train_model = Classification_Model.ModelTraining(num_epochs=40, batch_size=32, report_period=10)
models_copy, results_copy = train_model.trainModel(synthetic_dataset, decay_epochs=20)
accuracy_copy, cm_recall_copy = Result_Analysis.getAccuracyCm(results_copy)
# Confusion_Matrix.plotConfusionMatrix(cm_recall_noise, class_labels, normalize=False)
Storage.saveClassifyResult(subject, accuracy_copy, cm_recall_copy, version, result_set, 'classify_old_copy', gen_model, num_reference=5)

