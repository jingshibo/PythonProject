##
from Conditional_GAN.Models import cGAN_Training, cGAN_Testing, cGAN_Evaluation, Model_Storage
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data, Plot_Emg_Data, Dtw_Similarity
import datetime


'''train generative model'''
##  define windows
window_parameters = Process_Raw_Data.ganWindowParameters()
lower_limit = 20
higher_limit = 200
envelope_cutoff = 200
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
# modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'],
#     'emg_SALW': ['emg_SASA', 'emg_LWLW', 'emg_SALW'], 'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}
# modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD']}
modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA']}
# modes_generation = {'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD']}
time_interval = 5
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']
extracted_emg, train_gan_data = Process_Raw_Data.extractSeparateEmgData(modes_generation, old_emg_reshaped, new_emg_reshaped, time_interval,
    length, output_list=True)


## hyperparameters
num_epochs = 20
decay_epochs = [30, 45]
batch_size = 80
sampling_repetition = 100  # the number of samples for each time point
gen_update_interval = 3  # The frequency at which the generator is updated. if set to 2, the generator is updated every 2 batches.
disc_update_interval = 1  # The frequency at which the discriminator is updated. if set to 2, the discriminator is updated every 2 batches.
noise_dim = 64
blending_factor_dim = 2


## GAN data storage information
subject = 'Test'
version = 1  # the data from which experiment version to process
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
epoch_number = 30
gen_results = Model_Storage.loadBlendingFactors(subject, version, result_set, model_type, modes_generation, checkpoint_result_path, epoch_number=epoch_number)
test_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
synthetic_data = test_evaluation.generateFakeData(extracted_emg, 'old', modes_generation, old_emg_normalized, envelope_cutoff, repetition=1, random_pairing=False)


## screen representative fake data for classification model training
extracted_data = Dtw_Similarity.extractFakeData(synthetic_data, old_emg_normalized, modes_generation, cutoff_frequency=50, num_sample=60,
    num_reference=1, method='best')


## plot to see how the dtw plot looks like (test other envelope cutoff frequency, or use eular distance directly)
Dtw_Similarity.plotPath(extracted_data['fake_averaged'], extracted_data['real_averaged'], source='emg_1_repetition_list', mode='emg_LWSA',
    fake_index=extracted_data['selected_fake_index_1'][1], reference_index=extracted_data['selected_reference_index_1'][0])


## plotting fake and real emg data for comparison
fake_old_1 = Plot_Emg_Data.averageEmgValues(extracted_data['selected_fake_data_1'])
real_old = Plot_Emg_Data.averageEmgValues(old_emg_normalized)
# plot multiple locomotion mode emg in a single plot for comparison
old_to_plot_1 = {'fake_LWSA': fake_old_1['emg_1_event_mean']['emg_LWSA'], 'real_LWSA': real_old['emg_1_event_mean']['emg_LWSA'],
    'real_SASA': real_old['emg_1_event_mean']['emg_SASA'], 'real_LWLW': real_old['emg_1_event_mean']['emg_LWLW']}
Plot_Emg_Data.plotMultipleModeValues(old_to_plot_1, title='emg_1_on_1', ylim=(0, 0.5))
old_to_plot_2 = {'fake_LWSA': fake_old_1['emg_2_event_mean']['emg_LWSA'], 'real_LWSA': real_old['emg_2_event_mean']['emg_LWSA'],
    'real_SASA': real_old['emg_2_event_mean']['emg_SASA'], 'real_LWLW': real_old['emg_2_event_mean']['emg_LWLW']}
Plot_Emg_Data.plotMultipleModeValues(old_to_plot_2, title='emg_2_on_1', ylim=(0, 0.5))

# plot multiple locomotion mode emg in a single plot for comparison
fake_old_2 = Plot_Emg_Data.averageEmgValues(extracted_data['selected_fake_data_2'])
old_to_plot_1 = {'fake_LWSA': fake_old_2['emg_1_event_mean']['emg_LWSA'], 'real_LWSA': real_old['emg_1_event_mean']['emg_LWSA'],
    'real_SASA': real_old['emg_1_event_mean']['emg_SASA'], 'real_LWLW': real_old['emg_1_event_mean']['emg_LWLW']}
Plot_Emg_Data.plotMultipleModeValues(old_to_plot_1, title='emg_1_on_2', ylim=(0, 0.5))
old_to_plot_2 = {'fake_LWSA': fake_old_2['emg_2_event_mean']['emg_LWSA'], 'real_LWSA': real_old['emg_2_event_mean']['emg_LWSA'],
    'real_SASA': real_old['emg_2_event_mean']['emg_SASA'], 'real_LWLW': real_old['emg_2_event_mean']['emg_LWLW']}
Plot_Emg_Data.plotMultipleModeValues(old_to_plot_2, title='emg_2_on_2', ylim=(0, 0.5))
# # plot multiple emg values of each locomotion mode in subplots for comparison
# Plot_Emg_Data.plotAverageValue(fake_old['emg_1_repetition_list'], 'emg_LWSA', 2, title='fake_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_1_repetition_list'], 'emg_LWSA', 2, title='real_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_1_repetition_list'], 'emg_LWLW', 2, title='real_LWLW', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_1_repetition_list'], 'emg_SASA', 2, title='real_SASA', ylim=(0, 1))
# # plot the average psd of each locomotion mode for comparison
# Plot_Emg_Data.plotPsd(fake_old['emg_1_event_mean'], 'emg_LWSA',  list(range(30)), [6, 5], title='fake_LWSA')
# Plot_Emg_Data.plotPsd(real_old['emg_1_event_mean'], 'emg_LWSA', list(range(30)), [6, 5], title='real_LWSA')
# Plot_Emg_Data.plotPsd(real_old['emg_1_event_mean'], 'emg_LWLW', list(range(30)), [6, 5], title='real_LWLW')



##
# combine two curves into one
# fake_both = {}
# real_both = {}
# for transition_type in modes_generation.keys():
#     fake_both[transition_type] = np.concatenate((fake['emg_1_repetition_list'][transition_type], fake['emg_2_repetition_list'][transition_type]), axis=0)
#     real_both[transition_type] = np.concatenate((real['emg_1_repetition_list'][transition_type], real['emg_2_repetition_list'][transition_type]), axis=0)
# dtw_results = dtw_distance.calcuDtwDistance(fake_both, real_both)
# selected_fake_data, selected_fake_index, selected_reference_index = dtw_distance.selectFakeData(dtw_results, synthetic_data)
