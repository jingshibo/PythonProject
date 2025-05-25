##
from Conditional_GAN.Data_Procesing import Process_Raw_Data, Train_Classifiers, Train_cGan
from cGAN_Transformer.Functions import Preprocessing, Results, Storage
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
    down_up_session_t0, up_down_session_t1, down_up_session_t1, grid=grid, envelope=True, envelope_cutoff=400, reordering=False)

## parameters for extracting emg data to train models
amplitude_limit = 1500
spatial_filter_kernel = (2, 1)
modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'],
    'emg_SALW': ['emg_SASA', 'emg_LWLW', 'emg_SALW'],
    'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}  # The order in each list is important, corresponding to gen_data_1 and gen_data_2.
condition_encoding = {"emg_LWSA": 0, "emg_LWSD": 1, "emg_SALW": 2, "emg_SDLW": 3}  # encode the conditions into integer
num_conditions = len(condition_encoding)
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']  # the length of data in each repetition
old_emg_normalized, new_emg_normalized, _, _ = Process_Raw_Data.normalizeFilterEmgData(old_emg_data, new_emg_data, amplitude_limit,
    normalize='(0,1)', spatial_filter=False, kernel=spatial_filter_kernel)
time_range = [-200, 200]  # select data centered around toe-off
old_emg_central, new_emg_central, train_gan_data = Preprocessing.extractGanTrainingData(modes_generation, old_emg_normalized,
    new_emg_normalized, time_range)
classify_old_emg = Preprocessing.buildClassifyDataset(old_emg_central)


# ##
# import numpy as np
# import matplotlib.pyplot as plt
#
# # The 7 EMG data keys
# keys = ['emg_LWLW', 'emg_LWSA', 'emg_SASA', 'emg_SDSD', 'emg_SDLW', 'emg_LWSD', 'emg_SALW']
#
# # Setup subplots
# fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 14))
# axes = axes.flatten()
#
# for idx, key in enumerate(keys):
#     ax = axes[idx]
#     data_list = old_emg_central[key]
#
#     for i in range(min(30, len(data_list))):  # limit to 30 samples
#         sample = data_list[i]  # Shape: (T, 65)
#         avg_signal = np.mean(sample, axis=1)  # Mean across channels
#         ax.plot(avg_signal, alpha=0.6, label=f'Sample {i+1}')
#
#     ax.set_title(f'{key}')
#     ax.set_xlabel('Time Step')
#     ax.set_ylabel('Avg Channel Value')
#     ax.set_ylim(0, 0.8)  # Set y-axis limit
#     ax.grid(True)
#
# # Hide unused subplot if only using 7
# for j in range(len(keys), len(axes)):
#     fig.delaxes(axes[j])
#
# plt.tight_layout()
# plt.show()
#
#
# ##
# # Choose one key to plot
# key = 'emg_SASA'
# data_list = old_emg_central[key]
#
# # Setup: 6 rows × 5 cols = 30 subplots
# fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 15))
# axes = axes.flatten()
#
# for i in range(30):
#     sample = data_list[i]  # Shape: (T, 65)
#     avg_signal = np.mean(sample, axis=1)  # Shape: (T,)
#
#     ax = axes[i]
#     ax.plot(avg_signal)
#     ax.set_title(f'Sample {i+1}')
#     ax.set_ylim(0, 0.8)  # Set y-axis limit
#     ax.set_xlabel('Time Step')
#     ax.set_ylabel('Avg Value')
#     ax.grid(True)
#
# plt.tight_layout()
# plt.show()
#
# ##
# # Choose a key (e.g., 'emg_LWLW')
# key = 'emg_SASA'
# data_list = old_emg_central[key]
#
# # Setup 6×5 subplot grid
# fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 15))
# axes = axes.flatten()
#
# for i in range(30):
#     sample = data_list[i]  # Shape: (2000, 65)
#     matrix = sample.T       # Transpose to (channels, time) => (65, 2000)
#
#     ax = axes[i]
#     im = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=0, vmax=0.6)
#     ax.set_title(f'Sample {i+1}')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Channel')
#     ax.set_xticks([])
#     ax.set_yticks([])
#
# # Adjust layout and add colorbar
# plt.tight_layout()
# cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
# fig.colorbar(im, cax=cbar_ax, label='Signal Value')
#
# plt.show()

## train gan model
NUM_EPOCHS = 50
BATCH_SIZE = 32
SAMPLING_REPETITION = 50
model_type = 'Transformer_Gan'
model_name = ['gen', 'disc']
training_parameters = {'modes_generation': modes_generation, 'sampling_repetition': SAMPLING_REPETITION, 'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS}
storage_parameters = {'subject': subject, 'version': version, 'model_type': model_type, 'model_name': model_name, 'gan_result_set': 0}
# trainer = Transformer_GAN_Training.GanTraining(NUM_EPOCHS, BATCH_SIZE, SAMPLING_REPETITION, num_conditions)
# model = trainer.trainModel(train_gan_data, condition_encoding, training_parameters, storage_parameters)


# ## generate transition data
# epoch_number = 30
# model = Storage.loadCheckPointModels(storage_parameters, epoch_number)
# generated_transition_data = Transformer_GAN_Testing.generateTransitionData(model['gen'], train_gan_data, condition_encoding,
#     sample_number=30, batch_size=150)
#
# # --- The substitution logic ---
# old_emg = copy.deepcopy(old_emg_central)
# for key_gen, data_gen in generated_transition_data.items():
#     old_emg[key_gen] = [data_gen[i].squeeze(0).T for i in range(data_gen.shape[0])]  # Replace the list with the ndarray
# generated_old_emg = Preprocessing.buildClassifyDataset(old_emg_central)


## print image











##  classify using a single cnn 2d model
num_epochs = 50
batch_size = 32
decay_epochs = 20
now = datetime.datetime.now()
fold_number = 5
cross_validation_indices, cross_validation_dataset = Preprocessing.crossValidationSet(fold_number, classify_old_emg)
train_model = Classification_Model.ModelTraining(num_epochs, batch_size, report_period=10)
models, model_results = train_model.trainModel(classify_old_emg, cross_validation_indices, decay_epochs)
print(datetime.datetime.now() - now)
accuracy, cm_recall = Results.getAccuracyCm(model_results)
class_labels = ['LW', 'LWSA', 'LWSD', 'SALW', 'SA', 'SDLW', 'SD']
Confusion_Matrix.plotConfusionMatrix(cm_recall, class_labels, normalize=False)
