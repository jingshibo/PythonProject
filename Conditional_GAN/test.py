##
from Conditional_GAN.Models import cGAN_Training, cGAN_Testing, cGAN_Evaluation, Model_Storage
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data, Plot_Emg_Data
import datetime


'''train generative model'''
##  define windows
window_parameters = Process_Raw_Data.returnWindowParameters()
lower_limit = 20
higher_limit = 400
envelope_cutoff = 400
envelope = True  # the output will always be rectified if set True


## read and filter old data
subject = 'Number4'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 2, 3, 4]
down_up_session = [0, 1, 2, 5, 6]
# up_down_session = [0]
# down_up_session = [0]
sessions = [up_down_session, down_up_session]
data_source = {'subject': subject, 'version': version, 'modes': modes, 'sessions': sessions}
# old_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
#     envelope_cutoff=envelope_cutoff, envelope=envelope)
old_emg_data_classify = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    envelope_cutoff=envelope_cutoff, envelope=envelope)


## read and filter new data
subject = 'Number5'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [5, 6, 7, 8, 9]
down_up_session = [4, 5, 6, 8, 9]
# up_down_session = [0]
# down_up_session = [0]
sessions = [up_down_session, down_up_session]
data_source = {'subject': subject, 'version': version, 'modes': modes, 'sessions': sessions}
# new_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
#     envelope_cutoff=envelope_cutoff, envelope=envelope)
new_emg_data_classify = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    envelope_cutoff=envelope_cutoff, envelope=envelope)


## normalize and extract emg data for gan model training
range_limit = 2000
old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped = Process_Raw_Data.normalizeReshapeEmgData(old_emg_data_classify,
    new_emg_data_classify, range_limit, normalize='(0,1)')
# The order in each list is important, corresponding to gen_data_1 and gen_data_2.
# modes_generation = {'LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'], 'SALW': ['emg_SASA',
# 'emg_LWLW', 'emg_SALW'], 'SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}
# modes_generation = {'LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD']}
modes_generation = {'LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA']}
# modes_generation = {'LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD']}
time_interval = 5
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']
extracted_emg, train_gan_data = Process_Raw_Data.extractSeparateEmgData(modes_generation, old_emg_reshaped, new_emg_reshaped, time_interval,
    length, output_list=True)


## hyperparameters
num_epochs = 50
decay_epochs = [30, 45]
batch_size = 1000
sampling_repetition = 100  # the number of samples for each time point
gen_update_interval = 3  # The frequency at which the generator is updated. if set to 2, the generator is updated every 2 batches.
disc_update_interval = 1  # The frequency at which the discriminator is updated. if set to 2, the discriminator is updated every 2 batches.
noise_dim = 64
blending_factor_dim = 2


## GAN data storage information
subject = 'Test'
version = 0  # the data from which experiment version to process
checkpoint_model_path = f'D:\Data\cGAN_Model\subject_{subject}\Experiment_{version}\models\check_points'
checkpoint_result_path = f'D:\Data\cGAN_Model\subject_{subject}\Experiment_{version}\model_results\check_points'
model_type = 'cGAN'
model_name = ['gen', 'disc']
result_set = 0


## train and save gan models for multiple transitions
training_parameters = {'modes_generation': modes_generation, 'noise_dim': noise_dim, 'sampling_repetition': sampling_repetition,
    'gen_update_interval': gen_update_interval, 'disc_update_interval': disc_update_interval, 'batch_size': batch_size,
    'num_epochs': num_epochs, 'decay_epochs': decay_epochs, 'interval': time_interval, 'blending_factor_dim': blending_factor_dim}
storage_parameters = {'subject': subject, 'version': version, 'model_type': model_type, 'model_name': model_name, 'result_set': result_set,
    'checkpoint_model_path': checkpoint_model_path, 'checkpoint_result_path': checkpoint_result_path}
now = datetime.datetime.now()
results = {}
for transition_type in modes_generation.keys():
    gan_models, blending_factors = cGAN_Training.trainCGan(train_gan_data[transition_type], transition_type, training_parameters, storage_parameters)
    results[transition_type] = blending_factors
print(datetime.datetime.now() - now)


## test generated data results
epoch_number = 50
gen_results = Model_Storage.loadBlendingFactors(subject, version, result_set, model_type, modes_generation, checkpoint_result_path, epoch_number=epoch_number)
test_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
synthetic_data = test_evaluation.generateFakeData(extracted_emg, 'old', modes_generation, old_emg_normalized, envelope_cutoff, repetition=1, random_pairing=False)
# synthetic_data['emg_LWSA'] = synthetic_data['emg_LWSA'][0:60]


## DTW matching
import random
import numpy as np
import matplotlib.pyplot as plt
from dtw import accelerated_dtw

# low pass filtering
cutoff_frequency = 50
synthetic_envelope = {key: Process_Fake_Data.clipSmoothEmgData(value, cutoff_frequency) for key, value in synthetic_data}
old_emg_envelope = {key: Process_Fake_Data.clipSmoothEmgData(value, cutoff_frequency) for key, value in old_emg_normalized}
# calculate mean value across all channels for each repetition
fake = Plot_Emg_Data.averageEmgValues(synthetic_envelope)
real = Plot_Emg_Data.averageEmgValues(old_emg_envelope)
# randomly select a real LWSA data
reference_emg_1 = random.choice(real['emg_1_repetition_list']['emg_LWSA'])
reference_emg_2 = random.choice(real['emg_2_repetition_list']['emg_LWSA'])

# compute dtw distance between reference and synthetic data
dtw_distance = []
warp_paths = []
for fake_data in fake['emg_1_repetition_list']['emg_LWSA']:
    distance, cost_matrix, accumulated_cost_matrix, path = accelerated_dtw(reference_emg_1, fake_data, dist='euclidean')
    best_path = list(zip(path[0], path[1]))
    dtw_distance.append(distance)
    warp_paths.append(best_path)


# You can also visualise the accumulated cost and the shortest path
plt.imshow(accumulated_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()


## plotting fake and real emg data for comparison
# fake = Plot_Emg_Data.averageEmgValues(synthetic_data)
# real = Plot_Emg_Data.averageEmgValues(old_emg_normalized)
# # plot multiple emg values of each locomotion mode in subplots for comparison
# Plot_Emg_Data.plotAverageValue(fake['emg_2_repetition_list'], 'emg_LWSA', list(range(30)), [6, 5], title='fake_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real['emg_2_repetition_list'], 'emg_LWSA', list(range(30)), [6, 5], title='real_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real['emg_2_repetition_list'], 'emg_LWLW', list(range(30)), [6, 5], title='real_LWLW', ylim=(0, 1))
# # plot multiple locomotion mode emg in a single plot for comparison
# emg_to_plot = {'fake_LWSA': fake['emg_1_event_mean']['emg_LWSA'], 'real_LWSA': real['emg_1_event_mean']['emg_LWSA'],
#     'real_SASA': real['emg_1_event_mean']['emg_SASA'], 'real_LWLW': real['emg_1_event_mean']['emg_LWLW']}
# Plot_Emg_Data.plotMultipleModeValues(emg_to_plot)
# # plot the average psd of each locomotion mode for comparison
# Plot_Emg_Data.plotPsd(fake['emg_1_event_mean'], 'emg_LWSA',  list(range(30)), [6, 5], title='fake_LWSA')
# Plot_Emg_Data.plotPsd(real['emg_1_event_mean'], 'emg_LWSA', list(range(30)), [6, 5], title='real_LWSA')
# Plot_Emg_Data.plotPsd(real['emg_1_event_mean'], 'emg_LWLW', list(range(30)), [6, 5], title='real_LWLW')


