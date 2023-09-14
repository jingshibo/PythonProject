##
import copy

from Conditional_GAN.Models import cGAN_Training, cGAN_Testing, cGAN_Evaluation, Model_Storage
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data, Plot_Emg_Data, Dtw_Similarity
import datetime


'''train generative model'''
##  define windows
window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=450, endtime_after_toeoff_ms=400, feature_window_ms=450)
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


# read and filter new data
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
# old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped = Process_Raw_Data.normalizeReshapeEmgData(old_emg_data_classify,
#     new_emg_data_classify, range_limit, normalize='(0,1)')
# The order in each list is important, corresponding to gen_data_1 and gen_data_2.
# modes_generation = {'LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'], 'SALW': ['emg_SASA',
# 'emg_LWLW', 'emg_SALW'], 'SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}
# modes_generation = {'LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD']}
modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA']}
# modes_generation = {'LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD']}
time_interval = 5
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']
# extracted_emg, train_gan_data = Process_Raw_Data.extractSeparateEmgData(modes_generation, old_emg_reshaped, new_emg_reshaped, time_interval,
#     length, output_list=True)


# hyperparameters
num_epochs = 30
decay_epochs = [30, 45]
batch_size = 800  # maximum value to prevent overflow during running
sampling_repetition = 100  # the number of samples for each time point
gen_update_interval = 3  # The frequency at which the generator is updated. if set to 2, the generator is updated every 2 batches.
disc_update_interval = 1  # The frequency at which the discriminator is updated. if set to 2, the discriminator is updated every 2 batches.
noise_dim = 64
blending_factor_dim = 2


# GAN data storage information
subject = 'Test'
version = 3  # the data from which experiment version to process
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
# now = datetime.datetime.now()
# results = {}
# for transition_type in modes_generation.keys():
#     gan_models, blending_factors = cGAN_Training.trainCGan(train_gan_data[transition_type], transition_type, training_parameters, storage_parameters)
#     results[transition_type] = blending_factors
# print(datetime.datetime.now() - now)


# ## test generated data results
# epoch_number = 10
# gen_results = Model_Storage.loadBlendingFactors(subject, version, result_set, model_type, modes_generation, checkpoint_result_path, epoch_number=epoch_number)
# test_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
# synthetic_data = test_evaluation.generateFakeData(extracted_emg, 'old', modes_generation, old_emg_normalized, envelope_cutoff, repetition=1, random_pairing=False)
# synthetic_data['emg_LWSA'] = synthetic_data['emg_LWSA'][0:60]
#
#
# ## plotting fake and real emg data for comparison
# fake_old = Plot_Emg_Data.averageEmgValues(synthetic_data)
# real_old = Plot_Emg_Data.averageEmgValues(old_emg_normalized)
# # plot multiple locomotion mode emg in a single plot for comparison
# old_to_plot_1 = {'fake_LWSA': fake_old['emg_1_event_mean']['emg_LWSA'], 'real_LWSA': real_old['emg_1_event_mean']['emg_LWSA'],
#     'real_SASA': real_old['emg_1_event_mean']['emg_SASA'], 'real_LWLW': real_old['emg_1_event_mean']['emg_LWLW']}
# Plot_Emg_Data.plotMultipleModeValues(old_to_plot_1, title='old_data_1', ylim=(0, 0.5))
# old_to_plot_2 = {'fake_LWSA': fake_old['emg_2_event_mean']['emg_LWSA'], 'real_LWSA': real_old['emg_2_event_mean']['emg_LWSA'],
#     'real_SASA': real_old['emg_2_event_mean']['emg_SASA'], 'real_LWLW': real_old['emg_2_event_mean']['emg_LWLW']}
# Plot_Emg_Data.plotMultipleModeValues(old_to_plot_2, title='old_data_2', ylim=(0, 0.5))
# # plot multiple emg values of each locomotion mode in subplots for comparison
# Plot_Emg_Data.plotAverageValue(fake_old['emg_1_repetition_list'], 'emg_LWSA', 2, title='fake_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_1_repetition_list'], 'emg_LWSA', 2, title='real_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_1_repetition_list'], 'emg_LWLW', 2, title='real_LWLW', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_1_repetition_list'], 'emg_SASA', 2, title='real_SASA', ylim=(0, 1))
# # plot the average psd of each locomotion mode for comparison
# Plot_Emg_Data.plotPsd(fake_old['emg_1_event_mean'], 'emg_LWSA',  list(range(30)), [6, 5], title='fake_LWSA')
# Plot_Emg_Data.plotPsd(real_old['emg_1_event_mean'], 'emg_LWSA', list(range(30)), [6, 5], title='real_LWSA')
# Plot_Emg_Data.plotPsd(real_old['emg_1_event_mean'], 'emg_LWLW', list(range(30)), [6, 5], title='real_LWLW')




'''
    train classifier (on subject 4), for testing gan generation performance
'''
## laod training data
# load blending factors for each transition type to generate
epoch_number = 30
gen_results = Model_Storage.loadBlendingFactors(subject, version, result_set, model_type, modes_generation, checkpoint_result_path,
    epoch_number=epoch_number)
# normalize and extract emg data for classification model training
old_emg_classify_normalized, new_emg_classify_normalized, old_emg_classify_reshaped, new_emg_classify_reshaped = \
    Process_Raw_Data.normalizeReshapeEmgData(old_emg_data_classify, new_emg_data_classify, range_limit, normalize='(0,1)')
extracted_emg_classify, _ = Process_Raw_Data.extractSeparateEmgData(modes_generation, old_emg_classify_reshaped, new_emg_classify_reshaped,
    time_interval, length, output_list=False)


# generate fake data
old_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
synthetic_old_data = old_evaluation.generateFakeData(extracted_emg_classify, 'old', modes_generation, old_emg_classify_normalized,
    envelope_cutoff, repetition=1, random_pairing=True)  # repetition and random_pairing are two unnecessary parameters.
reordered_fake_old_data = Process_Fake_Data.reorderDataSet(synthetic_old_data)
reordered_real_old_data = Process_Fake_Data.reorderDataSet(old_emg_classify_normalized)


# selected given time range of data
import numpy as np
# train classifier
shorten_fake_emg = {}
start = 0
end = 850
# Loop through each key-value pair in the original dictionary
for key, array_list in reordered_fake_old_data.items():
    # Loop through each array in the list and select only the first 65 columns
    shorten_fake_emg[key] = [arr[start:end, :] for arr in array_list]

shorten_old_emg = {}
# Loop through each key-value pair in the original dictionary
for key, array_list in old_emg_classify_normalized.items():
    shorten_old_emg[key] = [arr[start:end, :] for arr in array_list]

window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=450-start, endtime_after_toeoff_ms=end-450, feature_window_ms=450-start)


# screen representative fake data for classification model training
extracted_old_data = Dtw_Similarity.extractFakeData(shorten_fake_emg, shorten_old_emg, modes_generation, cutoff_frequency=50, num_sample=60,
    num_reference=1, method='select')


# retain one emg
fake_emg_dict_1 = {}
fake_emg_dict_2 = {}
channel_number = 65
# Loop through each key-value pair in the original dictionary
for key, array_list in extracted_old_data['selected_fake_data_2'].items():
    # Loop through each array in the list and select only the first 65 columns
    fake_emg_dict_1[key] = [arr[:, :channel_number] for arr in array_list]
    fake_emg_dict_2[key] = [arr[:, channel_number:] for arr in array_list]
real_emg_dict_1 = {}
real_emg_dict_2 = {}
# Loop through each key-value pair in the original dictionary
for key, array_list in shorten_old_emg.items():
    # Loop through each array in the list and select only the first 65 columns
    real_emg_dict_1[key] = [arr[:, :channel_number] for arr in array_list]
    real_emg_dict_2[key] = [arr[:, channel_number:] for arr in array_list]


# median filtering
import pandas as pd
import copy
from scipy import ndimage
filtered_fake_dict = copy.deepcopy(fake_emg_dict_2)
# Loop through each key-value pair in the original dictionary
for key, array_list in fake_emg_dict_2.items():
    # if key in modes_generation.keys():
    # Loop through each array in the list and apply median filtering
    filtered_array_list = [pd.DataFrame(ndimage.median_filter(arr, mode='nearest', size=3)).to_numpy() for arr in array_list]
    # Add the new list of filtered arrays to the new dictionary
    filtered_fake_dict[key] = filtered_array_list

filtered_real_dict = {}
# Loop through each key-value pair in the original dictionary
for key, array_list in real_emg_dict_2.items():
    # Loop through each array in the list and apply median filtering
    filtered_array_list = [pd.DataFrame(ndimage.median_filter(arr, mode='nearest', size=5)).to_numpy() for arr in array_list]
    # Add the new list of filtered arrays to the new dictionary
    filtered_real_dict[key] = filtered_array_list


# construct reference data and
import numpy as np
reference_data = {'emg_LWSA': [real_emg_dict_2['emg_LWSA'][index] for index in extracted_old_data['selected_reference_index_2']['emg_LWSA']]}
reference_data['emg_LWSA'] = [np.concatenate([arr, arr], axis=1) for arr in reference_data['emg_LWSA']]


#
old_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
train_set, shuffled_train_set = old_evaluation.classifierTrainSet(filtered_fake_dict, training_percent=0.8)
models_old, model_result_old = old_evaluation.trainClassifier(shuffled_train_set)
acc_old, cm_old = old_evaluation.evaluateClassifyResults(model_result_old)
# test classifier
shuffled_test_set = old_evaluation.classifierTestSet(modes_generation, filtered_real_dict, train_set, test_ratio=0.5)
test_results = old_evaluation.testClassifier(models_old[0], shuffled_test_set)
accuracy_old, cm_recall_old = old_evaluation.evaluateClassifyResults(test_results)



## plot to see how the dtw plot looks like (test other envelope cutoff frequency, or use eular distance directly)
# Dtw_Similarity.plotPath(extracted_old_data['fake_averaged'], extracted_old_data['real_averaged'], source='emg_1_repetition_list',
#     mode='emg_LWSA', fake_index=extracted_old_data['selected_fake_index_1'][1], reference_index=extracted_old_data['selected_reference_index_1'][0])


## plotting fake and real emg data for comparison
fake_old_1 = Plot_Emg_Data.averageEmgValues(extracted_old_data['selected_fake_data_1'])
fake_old_2 = Plot_Emg_Data.averageEmgValues(extracted_old_data['selected_fake_data_2'])
real_old = Plot_Emg_Data.averageEmgValues(shorten_old_emg)
reference = Plot_Emg_Data.averageEmgValues(reference_data)
## plot multiple locomotion mode emg in a single plot for comparison
# old_to_plot_1 = {'fake_LWSA': fake_old_1['emg_1_event_mean']['emg_LWSA'], 'real_LWSA': real_old['emg_1_event_mean']['emg_LWSA'],
#     'real_SASA': real_old['emg_1_event_mean']['emg_SASA'], 'real_LWLW': real_old['emg_1_event_mean']['emg_LWLW'], 'real_LWSS': real_old['emg_1_event_mean']['emg_LWSS']}
# Plot_Emg_Data.plotMultipleModeValues(old_to_plot_1, title='emg_1_on_1', ylim=(0, 0.5))
# old_to_plot_2 = {'fake_LWSA': fake_old_1['emg_2_event_mean']['emg_LWSA'], 'real_LWSA': real_old['emg_2_event_mean']['emg_LWSA'],
#     'real_SASA': real_old['emg_2_event_mean']['emg_SASA'], 'real_LWLW': real_old['emg_2_event_mean']['emg_LWLW'], 'real_LWSS': real_old['emg_2_event_mean']['emg_LWSS']}
# Plot_Emg_Data.plotMultipleModeValues(old_to_plot_2, title='emg_2_on_1', ylim=(0, 0.5))

# plot multiple locomotion mode emg in a single plot for comparison
# old_to_plot_1 = {'fake_LWSA': fake_old_2['emg_1_event_mean']['emg_LWSA'], 'real_LWSA': real_old['emg_1_event_mean']['emg_LWSA'],
#     'real_SASA': real_old['emg_1_event_mean']['emg_SASA'], 'real_LWLW': real_old['emg_1_event_mean']['emg_LWLW'], 'real_LWSS': real_old['emg_1_event_mean']['emg_LWSS']}
# Plot_Emg_Data.plotMultipleModeValues(old_to_plot_1, title='emg_1_on_2', ylim=(0, 0.5))
old_to_plot_2 = {'fake_LWSA': fake_old_2['emg_2_event_mean']['emg_LWSA'], 'real_LWSA': real_old['emg_2_event_mean']['emg_LWSA'],
    'real_SASA': real_old['emg_2_event_mean']['emg_SASA'], 'real_LWLW': real_old['emg_2_event_mean']['emg_LWLW'], 'real_LWSS': real_old['emg_2_event_mean']['emg_LWSS']}
Plot_Emg_Data.plotMultipleModeValues(old_to_plot_2, title='emg_2_on_2', ylim=(0, 0.5))


# ## plot multiple repetition values of each locomotion mode in subplots for comparison
# Plot_Emg_Data.plotAverageValue(fake_old_2['emg_2_repetition_list'], 'emg_LWSA', num_columns=30, title='fake_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_2_repetition_list'], 'emg_LWSA', num_columns=30, title='real_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_2_repetition_list'], 'emg_LWLW', num_columns=30, title='real_LWLW', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_2_repetition_list'], 'emg_LWSS', num_columns=30, title='real_LWSS', ylim=(0, 1))
# reference = Plot_Emg_Data.averageEmgValues(reference_data)
# Plot_Emg_Data.plotAverageValue(reference['emg_2_repetition_list'], 'emg_LWSA', num_columns=30, title='reference_LWSA', ylim=(0, 1))
#
#
# ## plot multiple channel values of each locomotion mode in subplots for comparison
# Plot_Emg_Data.plotAverageValue(fake_old_2['emg_2_channel_list'], 'emg_LWSA', num_columns=30, title='fake_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_2_channel_list'], 'emg_LWSA', num_columns=30, title='real_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_2_channel_list'], 'emg_LWLW', num_columns=30, title='real_LWLW', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_old['emg_2_channel_list'], 'emg_LWSS', num_columns=30, title='real_LWSS', ylim=(0, 1))
# reference = Plot_Emg_Data.averageEmgValues(reference_data)
# Plot_Emg_Data.plotAverageValue(reference['emg_2_channel_list'], 'emg_LWSA', num_columns=30, title='reference_LWSA', ylim=(0, 1))


## plot the average psd of each locomotion mode for comparison
Plot_Emg_Data.plotPsd(fake_old_2['emg_2_event_mean'], 'emg_LWSA', num_columns=30, title='fake_LWSA')
Plot_Emg_Data.plotPsd(real_old['emg_2_event_mean'], 'emg_LWSA', num_columns=30, title='real_LWSA')
Plot_Emg_Data.plotPsd(real_old['emg_2_event_mean'], 'emg_LWLW', num_columns=30, title='real_LWLW')
Plot_Emg_Data.plotPsd(real_old['emg_2_event_mean'], 'emg_LWSS', num_columns=30, title='real_LWSS')

Plot_Emg_Data.plotPsd(reference['emg_2_event_mean'], 'emg_LWSA', num_columns=30, title='real_LWSA')













## save results
# model_type = 'classify_old'
# Model_Storage.saveClassifyAccuracy(subject, accuracy_old, cm_recall_old, version, result_set, model_type, project='cGAN_Model')
# acc_old, cm_old = Model_Storage.loadClassifyAccuracy(subject, version, result_set, model_type, project='cGAN_Model')
# Model_Storage.saveClassifyModel(models_old[0], subject, version, model_type, project='cGAN_Model')
# model_old = Model_Storage.loadClassifyModel(subject, version, model_type, project='cGAN_Model')




'''
    train classifier (on subject 5), for evaluating the proposed method performance
'''
## generate fake data
new_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
synthetic_new_data = new_evaluation.generateFakeData(extracted_emg_classify, 'new', modes_generation, new_emg_classify_normalized,
    envelope_cutoff, repetition=1, random_pairing=False)

## screen representative fake data for classification model training
# low pass filtering
cutoff_frequency = 50
synthetic_envelope = {key: Process_Fake_Data.clipSmoothEmgData(value, cutoff_frequency) for key, value in synthetic_new_data.items()}
old_emg_envelope = {key: Process_Fake_Data.clipSmoothEmgData(value, cutoff_frequency) for key, value in old_emg_classify_normalized.items()}
# calculate mean value across all channels for each repetition
fake = Plot_Emg_Data.averageEmgValues(synthetic_envelope)
real = Plot_Emg_Data.averageEmgValues(old_emg_envelope)
# compute dtw distance between fake data and references
dtw_distance = Dtw_Similarity.Dtw_Distance(modes_generation, num_sample=30, num_reference=1)
dtw_results_1 = dtw_distance.calcuDtwDistance(fake['emg_1_repetition_list'], real['emg_1_repetition_list'])
dtw_results_2 = dtw_distance.calcuDtwDistance(fake['emg_2_repetition_list'], real['emg_2_repetition_list'])
selected_fake_data_1, selected_fake_index_1, selected_reference_index_1 = dtw_distance.selectFakeData(dtw_results_1, synthetic_new_data)
selected_fake_data_2, selected_fake_index_2, selected_reference_index_2 = dtw_distance.selectFakeData(dtw_results_2, synthetic_new_data)



## plotting fake and real emg data for comparison
fake_new = Plot_Emg_Data.averageEmgValues(synthetic_new_data)
real_new = Plot_Emg_Data.averageEmgValues(new_emg_normalized)
# plot multiple locomotion mode emg in a single plot for comparison
new_to_plot_1 = {'fake_LWSA': fake_new['emg_1_event_mean']['emg_LWSA'], 'real_LWSA': real_new['emg_1_event_mean']['emg_LWSA'],
    'real_SASA': real_new['emg_1_event_mean']['emg_SASA'], 'real_LWLW': real_new['emg_1_event_mean']['emg_LWLW']}
Plot_Emg_Data.plotMultipleModeValues(new_to_plot_1, title='new_data_1')
new_to_plot_2 = {'fake_LWSA': fake_new['emg_2_event_mean']['emg_LWSA'], 'real_LWSA': real_new['emg_2_event_mean']['emg_LWSA'],
    'real_SASA': real_new['emg_2_event_mean']['emg_SASA'], 'real_LWLW': real_new['emg_2_event_mean']['emg_LWLW']}
Plot_Emg_Data.plotMultipleModeValues(new_to_plot_2, title='new_data_2')
# # plot multiple emg values of each locomotion mode in subplots for comparison
# Plot_Emg_Data.plotAverageValue(fake_new['emg_1_repetition_list'], 'emg_LWSA', 30, title='fake_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_new['emg_1_repetition_list'], 'emg_LWSA', 30, title='real_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_new['emg_1_repetition_list'], 'emg_LWLW', 30, title='real_LWLW', ylim=(0, 1))
# # plot the average psd of each locomotion mode for comparison
# Plot_Emg_Data.plotPsd(fake_new['emg_1_event_mean'], 'emg_LWSA',  list(range(30)), [6, 5], title='fake_LWSA')
# Plot_Emg_Data.plotPsd(real_new['emg_1_event_mean'], 'emg_LWSA', list(range(30)), [6, 5], title='real_LWSA')
# Plot_Emg_Data.plotPsd(real_new['emg_1_event_mean'], 'emg_LWLW', list(range(30)), [6, 5], title='real_LWLW')


## train classifier
train_set, shuffled_train_set = new_evaluation.classifierTrainSet(selected_fake_data_1, training_percent=0.8)
models_new, model_results_new = new_evaluation.trainClassifier(shuffled_train_set)
# acc_new, cm_new = new_evaluation.evaluateClassifyResults(model_results_new)
# test classifier
shuffled_test_set = new_evaluation.classifierTestSet(modes_generation, new_emg_classify_normalized, train_set, test_ratio=0.5)
test_results = new_evaluation.testClassifier(models_new[0], shuffled_test_set)
accuracy_new, cm_recall_new = new_evaluation.evaluateClassifyResults(test_results)


## save results
# model_type = 'classify_new'
# Model_Storage.saveClassifyAccuracy(subject, accuracy_new, cm_recall_new, version, result_set, model_type, project='cGAN_Model')
# acc_new, cm_new = Model_Storage.loadClassifyAccuracy(subject, version, result_set, model_type, project='cGAN_Model')
# Model_Storage.saveClassifyModel(models_new[0], subject, version, model_type, project='cGAN_Model')
# model_new = Model_Storage.loadClassifyModel(subject, version, model_type, project='cGAN_Model')




'''
    train classifier (on subject 5), for comparison purpose
'''
## build training data
old_emg_for_replacement = {modes[2]: old_emg_classify_normalized[modes[2]] for transition_type, modes in modes_generation.items()}
mix_old_new_data = Process_Fake_Data.replaceUsingFakeEmg(old_emg_for_replacement, new_emg_classify_normalized)


## plotting old and new emg data for comparison
mix_old_new = Plot_Emg_Data.averageEmgValues(mix_old_new_data)
real_new = Plot_Emg_Data.averageEmgValues(new_emg_normalized)
# plot multiple locomotion mode emg in a single plot for comparison
mix_to_plot_1 = {'old_LWSA': mix_old_new['emg_1_event_mean']['emg_LWSA'], 'new_LWSA': real_new['emg_1_event_mean']['emg_LWSA'],
    'new_SASA': real_new['emg_1_event_mean']['emg_SASA'], 'new_LWLW': real_new['emg_1_event_mean']['emg_LWLW']}
Plot_Emg_Data.plotMultipleModeValues(mix_to_plot_1, title='mix_data_1')
mix_to_plot_2 = {'old_LWSA': mix_old_new['emg_2_event_mean']['emg_LWSA'], 'new_LWSA': real_new['emg_2_event_mean']['emg_LWSA'],
    'new_SASA': real_new['emg_2_event_mean']['emg_SASA'], 'new_LWLW': real_new['emg_2_event_mean']['emg_LWLW']}
Plot_Emg_Data.plotMultipleModeValues(mix_to_plot_2, title='mix_data_2')
# # plot multiple emg values of each locomotion mode in subplots for comparison
# Plot_Emg_Data.plotAverageValue(mix_old_new['emg_1_repetition_list'], 'emg_LWSA', 30, title='fake_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_new['emg_1_repetition_list'], 'emg_LWSA', 30, title='real_LWSA', ylim=(0, 1))
# Plot_Emg_Data.plotAverageValue(real_new['emg_1_repetition_list'], 'emg_LWLW', 30, title='real_LWLW', ylim=(0, 1))
# # plot the average psd of each locomotion mode for comparison
# Plot_Emg_Data.plotPsd(mix_old_new['emg_1_event_mean'], 'emg_LWSA',  list(range(30)), [6, 5], title='fake_LWSA')
# Plot_Emg_Data.plotPsd(real_new['emg_1_event_mean'], 'emg_LWSA', list(range(30)), [6, 5], title='real_LWSA')
# Plot_Emg_Data.plotPsd(real_new['emg_1_event_mean'], 'emg_LWLW', list(range(30)), [6, 5], title='real_LWLW')


## train classifier
train_set, shuffled_train_set = new_evaluation.classifierTrainSet(mix_old_new_data, training_percent=0.8)
models_compare, model_results_compare = new_evaluation.trainClassifier(shuffled_train_set)
# acc_compare, cm_compare = new_evaluation.evaluateClassifyResults(model_results_compare)
# test classifier
shuffled_test_set = new_evaluation.classifierTestSet(modes_generation, new_emg_classify_normalized, train_set, test_ratio=0.5)
test_results = new_evaluation.testClassifier(models_compare[0], shuffled_test_set)
accuracy_compare, cm_recall_compare = new_evaluation.evaluateClassifyResults(test_results)


## save results
# model_type = 'classify_compare'
# Model_Storage.saveClassifyAccuracy(subject, accuracy_compare, cm_recall_compare, version, result_set, model_type, project='cGAN_Model')
# acc_compare, cm_compare = Model_Storage.loadClassifyAccuracy(subject, version, result_set, model_type, project='cGAN_Model')
# Model_Storage.saveClassifyModel(models_compare[0], subject, version, model_type, project='cGAN_Model')
# model_compare = Model_Storage.loadClassifyModel(subject, version, model_type, project='cGAN_Model')





## load check point models
output = {}
for transition_type in modes_generation.keys():
    test_model = Model_Storage.loadCheckPointModels(checkpoint_model_path, model_name, epoch_number=30, transition_type=transition_type)
    gen_model = cGAN_Testing.ModelTesting(test_model['gen'])
    result = gen_model.estimateBlendingFactors(train_gan_data[transition_type], noise_dim=noise_dim)
    output[transition_type] = result


