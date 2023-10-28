'''
    using conditional gan to generate transitional data and train the classifier from scratch based on sufficient new data
'''

##
import copy
import datetime
from Conditional_GAN.Models import cGAN_Training, cGAN_Testing, cGAN_Evaluation, Model_Storage
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data, Plot_Emg_Data, Dtw_Similarity, Post_Process_Data


'''train generative model'''
##  define windows
window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=450, endtime_after_toeoff_ms=400, feature_window_ms=450)
lower_limit = 20
higher_limit = 400
envelope_cutoff = 400
envelope = True  # the output will always be rectified if set True


## read and filter old data
subject = 'Number1'
version = 0  # the data from which experiment version to process
modes = ['up_down_t0', 'down_up_t0']
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [0, 1, 2, 5, 6]
# up_down_session = [5, 6, 7, 8, 9]
# down_up_session = [7, 8, 9, 10]
up_down_session = [0, 1, 2, 3, 4]
down_up_session = [1, 2, 3, 4, 5]
sessions = [up_down_session, down_up_session]
data_source = {'subject': subject, 'version': version, 'modes': modes, 'sessions': sessions}
# old_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
#     envelope_cutoff=envelope_cutoff, envelope=envelope)
old_emg_data_classify = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    envelope_cutoff=envelope_cutoff, envelope=envelope, project='cGAN_Model', selected_grid='grid_1')


# read and filter new data
modes = ['up_down_t1', 'down_up_t1']
# up_down_session = [5, 6, 7, 8, 9]
# down_up_session = [4, 5, 6, 8, 9]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [0, 1, 2, 3]
up_down_session = [0, 1, 2, 3, 4]
down_up_session = [1, 2, 3, 4, 5]
sessions = [up_down_session, down_up_session]
data_source = {'subject': subject, 'version': version, 'modes': modes, 'sessions': sessions}
# new_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
#     envelope_cutoff=envelope_cutoff, envelope=envelope)
new_emg_data_classify = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    envelope_cutoff=envelope_cutoff, envelope=envelope, project='cGAN_Model', selected_grid='grid_1')


## normalize and extract emg data for gan model training
range_limit = 2000
old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped = Process_Raw_Data.normalizeFilterEmgData(old_emg_data_classify,
    new_emg_data_classify, range_limit, normalize='(0,1)', spatial_filter=True, sigma=1, axes=(2, 3), radius=4)
# The order in each list is important, corresponding to gen_data_1 and gen_data_2.
modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'],
    'emg_SALW': ['emg_SASA', 'emg_LWLW', 'emg_SALW'], 'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}
# modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD']}
# modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA']}
# modes_generation = {'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD']}
time_interval = 5
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']
extracted_emg, train_gan_data = Process_Raw_Data.extractSeparateEmgData(modes_generation, old_emg_reshaped, new_emg_reshaped, time_interval,
    length, output_list=True)


## hyperparameters
num_epochs = 30
decay_epochs = [30, 45]
batch_size = 1024  # maximum value to prevent overflow during running
sampling_repetition = 100  # the number of samples for each time point
gen_update_interval = 3  # The frequency at which the generator is updated. if set to 2, the generator is updated every 2 batches.
disc_update_interval = 1  # The frequency at which the discriminator is updated. if set to 2, the discriminator is updated every 2 batches.
noise_dim = 64
blending_factor_dim = 0


# GAN data storage information
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
del old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped, extracted_emg, train_gan_data



'''
    load data for classifier training
'''
## laod training data
# load blending factors for each transition type to generate
epoch_number = None
gen_results = Model_Storage.loadBlendingFactors(subject, version, result_set, model_type, modes_generation, checkpoint_result_path,
    epoch_number=epoch_number)
# # Create a new dictionary with the modified ndarrays
# new_gen_results = {}
# for outer_key, inner_dict in gen_results.items():
#     new_model_results = {sub_key: arr[:, :, :, 5:10] for sub_key, arr in inner_dict['model_results'].items()}
#     new_gen_results[outer_key] = {
#         'model_results': new_model_results,
#         'training_parameters': inner_dict['training_parameters']
#     }
# gen_results = new_gen_results

# normalize and extract emg data for classification model training
old_emg_classify_normalized, new_emg_classify_normalized, old_emg_classify_reshaped, new_emg_classify_reshaped = \
    Process_Raw_Data.normalizeFilterEmgData(
    old_emg_data_classify, new_emg_data_classify, range_limit, normalize='(0,1)', spatial_filter=True, sigma=1, axes=(2, 3), radius=4)
extracted_emg_classify, _ = Process_Raw_Data.extractSeparateEmgData(modes_generation, old_emg_classify_reshaped, new_emg_classify_reshaped,
    time_interval, length, output_list=False)


'''
    train classifier (on old data), for testing gan generation performance
'''
## generate fake data
window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=450, endtime_after_toeoff_ms=400, feature_window_ms=450)
old_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
synthetic_old_data, fake_old_images = old_evaluation.generateFakeData(extracted_emg_classify, 'old', modes_generation, old_emg_classify_normalized,
    spatial_filtering=True, sigma=1, axes=(2, 3), radius=4)
# separate and store grids in a list if only use one grid later
old_fake_emg_grids = Post_Process_Data.separateEmgGrids(synthetic_old_data, separate=True)
old_real_emg_grids = Post_Process_Data.separateEmgGrids(old_emg_classify_normalized, separate=True)


## only preprocess selected grid and define time range of data
processed_old_fake_data = Process_Fake_Data.reorderSmoothDataSet(old_fake_emg_grids['grid_1'], filtering=False, modes=modes_generation)
processed_old_real_data = Process_Fake_Data.reorderSmoothDataSet(old_real_emg_grids['grid_1'], filtering=False, modes=None)
sliced_old_fake_data, window_parameters = Post_Process_Data.sliceTimePeriod(processed_old_fake_data, start=50, end=750)
sliced_old_real_data, _ = Post_Process_Data.sliceTimePeriod(processed_old_real_data, start=50, end=750)


## select representative fake data for classification model training
selected_old_fake_data = Dtw_Similarity.extractFakeData(sliced_old_fake_data, sliced_old_real_data, modes_generation, envelope_frequency=50,
    num_sample=60, num_reference=1, method='select', random_reference=False, split_grids=True) # 50Hz remove huge oscillation while maintain some extent variance
# median filtering
filtered_old_fake_data = Post_Process_Data.spatialFilterModelInput(selected_old_fake_data['fake_data_based_on_grid_1'], kernel=3)
filtered_old_real_data = Post_Process_Data.spatialFilterModelInput(sliced_old_real_data, kernel=5)
del old_fake_emg_grids, old_real_emg_grids, processed_old_fake_data, processed_old_real_data, sliced_old_fake_data, sliced_old_real_data


## classification
old_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
train_set, shuffled_train_set = old_evaluation.classifierTrainSet(filtered_old_fake_data, dataset='cross_validation_set')
models_old, model_result_old = old_evaluation.trainClassifier(shuffled_train_set)
acc_old, cm_old = old_evaluation.evaluateClassifyResultsByGroup(model_result_old)
# test classifier
test_set, shuffled_test_set = old_evaluation.classifierTestSet(modes_generation, filtered_old_real_data, train_set, test_ratio=0.5)
test_results = old_evaluation.testClassifier(models_old, shuffled_test_set)
accuracy_old, cm_recall_old = old_evaluation.evaluateClassifyResultsByGroup(test_results)


## plotting fake and real emg data for comparison
transition_type = 'emg_SALW'
modes = modes_generation[transition_type]
# calculate average values
# reference_old_data = {transition_type: [sliced_old_real_data[transition_type][index] for index in
#     selected_old_fake_data['reference_index_based_on_grid_1'][transition_type]]}
fake_old = Plot_Emg_Data.calcuAverageEmgValues(filtered_old_fake_data)
real_old = Plot_Emg_Data.calcuAverageEmgValues(filtered_old_real_data)
# reference_old = Plot_Emg_Data.averageEmgValues(reference_old_data)
# plot values of certain transition type
Plot_Emg_Data.plotMultipleEventMeanValues(fake_old, real_old, modes, title='old_emg_2_on_2', ylim=(0, 0.5))
# Plot_Emg_Data.plotMultipleRepetitionValues(fake_old, real_old, reference_old, modes, ylim=(0, 1))
# Plot_Emg_Data.plotMultipleChannelValues(fake_old, real_old, reference_old, modes, ylim=(0, 1))
# Plot_Emg_Data.plotMutipleEventPsdMeanValues(fake_old, real_old, reference_old, modes, ylim=(0, 0.1), grid='grid_1')
# # plot the dtw distance curve (two selected)
# old_fake_envelopes = selected_old_fake_data['fake_envelope_averaged']['emg_repetition_list']['grid_1']['emg_LWSA']
# old_real_envelopes = selected_old_fake_data['real_envelope_averaged']['emg_repetition_list']['grid_1']['emg_LWSA']
# old_fake_envelope_index = selected_old_fake_data['fake_index_based_on_grid_1']['emg_LWSA'][1]
# old_real_envelope_index = selected_old_fake_data['reference_index_based_on_grid_1']['emg_LWSA'][0]
# Dtw_Similarity.plotDtwPath(old_fake_envelopes, old_real_envelopes, old_fake_envelope_index, old_real_envelope_index)


## save results
model_type = 'classify_old'
Model_Storage.saveClassifyResult(subject, accuracy_old, cm_recall_old, version, result_set, model_type, project='cGAN_Model')
accuracy_old, cm_recall_old = Model_Storage.loadClassifyResult(subject, version, result_set, model_type, project='cGAN_Model')
Model_Storage.saveClassifyModels(models_old[0], subject, version, model_type, project='cGAN_Model')  # only save one model as an example here
model_old = Model_Storage.loadClassifyModels(subject, version, model_type, project='cGAN_Model')




'''
    train classifier (on new data), for evaluating the proposed method performance
'''
## generate fake data
window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=450, endtime_after_toeoff_ms=400, feature_window_ms=450)
new_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
synthetic_new_data, fake_new_images = new_evaluation.generateFakeData(extracted_emg_classify, 'new', modes_generation, new_emg_classify_normalized,
    spatial_filtering=True, sigma=1, axes=(2, 3), radius=4)
# separate and store grids in a list if only use one grid later
new_fake_emg_grids = Post_Process_Data.separateEmgGrids(synthetic_new_data, separate=True)
new_real_emg_grids = Post_Process_Data.separateEmgGrids(new_emg_classify_normalized, separate=True)


# only preprocess selected grid and define time range of data
processed_new_fake_data = Process_Fake_Data.reorderSmoothDataSet(new_fake_emg_grids['grid_1'], filtering=False, modes=modes_generation)
processed_new_real_data = Process_Fake_Data.reorderSmoothDataSet(new_real_emg_grids['grid_1'], filtering=False, modes=None)
sliced_new_fake_data, window_parameters = Post_Process_Data.sliceTimePeriod(processed_new_fake_data, start=50, end=750)
sliced_new_real_data, _ = Post_Process_Data.sliceTimePeriod(processed_new_real_data, start=50, end=750)


## select representative fake data for classification model training
selected_new_fake_data = Dtw_Similarity.extractFakeData(sliced_new_fake_data, sliced_new_real_data, modes_generation, envelope_frequency=50,
    num_sample=60, num_reference=1, method='select', random_reference=False, split_grids=True) # 50Hz remove huge oscillation while maintain some extent variance
# median filtering
filtered_new_fake_data = Post_Process_Data.spatialFilterModelInput(selected_new_fake_data['fake_data_based_on_grid_1'], kernel=3)
filtered_new_real_data = Post_Process_Data.spatialFilterModelInput(sliced_new_real_data, kernel=5)
del new_fake_emg_grids, new_real_emg_grids, processed_new_fake_data, processed_new_real_data, sliced_new_fake_data, sliced_new_real_data


# train classifier
new_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
train_set, shuffled_train_set = new_evaluation.classifierTrainSet(filtered_new_fake_data, dataset='cross_validation_set')
models_new, model_results_new = new_evaluation.trainClassifier(shuffled_train_set)
acc_new, cm_new = new_evaluation.evaluateClassifyResultsByGroup(model_results_new)
# test classifier
test_set, shuffled_test_set = new_evaluation.classifierTestSet(modes_generation, filtered_new_real_data, train_set, test_ratio=0.5)
test_results = new_evaluation.testClassifier(models_new, shuffled_test_set)
accuracy_new, cm_recall_new = new_evaluation.evaluateClassifyResultsByGroup(test_results)


## plotting fake and real emg data for comparison
transition_type = 'emg_SDLW'
modes = modes_generation[transition_type]
# calculate average values
# reference_new_data = {transition_type: [sliced_new_real_data[transition_type][index] for index in
#     selected_new_fake_data['reference_index_based_on_grid_1'][transition_type]]}
fake_new = Plot_Emg_Data.calcuAverageEmgValues(filtered_new_fake_data)
real_new = Plot_Emg_Data.calcuAverageEmgValues(filtered_new_real_data)
# reference_new = Plot_Emg_Data.averageEmgValues(reference_new_data)
# plot values of certain transition type
Plot_Emg_Data.plotMultipleEventMeanValues(fake_new, real_new, modes, title='new_emg_2_on_2', ylim=(0, 0.5))
# Plot_Emg_Data.plotMultipleRepetitionValues(fake_new, real_new, reference_new, modes, ylim=(0, 1))
# Plot_Emg_Data.plotMultipleChannelValues(fake_new, real_new, reference_new, modes, ylim=(0, 1))
# Plot_Emg_Data.plotMutipleEventPsdMeanValues(fake_new, real_new, reference_new, modes, ylim=(0, 0.1), grid='grid_1')
# # plot the dtw distance curve (between a selected fake data and reference data)
# new_fake_envelopes = selected_new_fake_data['fake_envelope_averaged']['emg_repetition_list']['grid_1']['emg_LWSA']
# new_real_envelopes = selected_new_fake_data['real_envelope_averaged']['emg_repetition_list']['grid_1']['emg_LWSA']
# new_fake_envelope_index = selected_new_fake_data['fake_index_based_on_grid_1']['emg_LWSA'][1]
# new_real_envelope_index = selected_new_fake_data['reference_index_based_on_grid_1']['emg_LWSA'][0]
# Dtw_Similarity.plotDtwPath(new_fake_envelopes, new_real_envelopes, new_fake_envelope_index, new_real_envelope_index)


## save results
model_type = 'classify_new'
Model_Storage.saveClassifyResult(subject, accuracy_new, cm_recall_new, version, result_set, model_type, project='cGAN_Model')
accuracy_new, cm_recall_new = Model_Storage.loadClassifyResult(subject, version, result_set, model_type, project='cGAN_Model')
Model_Storage.saveClassifyModels(models_new[0], subject, version, model_type, project='cGAN_Model')
model_new = Model_Storage.loadClassifyModels(subject, version, model_type, project='cGAN_Model')



'''
    train classifier (on new data), for comparison purpose
'''
## build training data
old_emg_for_replacement = {modes[2]: old_emg_classify_normalized[modes[2]] for transition_type, modes in modes_generation.items()}
mix_old_new_data = Process_Fake_Data.replaceUsingFakeEmg(old_emg_for_replacement, new_emg_classify_normalized)
# separate and store grids in a list if only use one grid later
mix_old_emg_grids = Post_Process_Data.separateEmgGrids(mix_old_new_data, separate=True)
new_real_emg_grids = Post_Process_Data.separateEmgGrids(new_emg_classify_normalized, separate=True)


## only preprocess selected grid and define time range of data
processed_mix_data = Process_Fake_Data.reorderSmoothDataSet(mix_old_emg_grids['grid_1'], filtering=False, modes=modes_generation)
processed_new_real_data = Process_Fake_Data.reorderSmoothDataSet(new_real_emg_grids['grid_1'], filtering=False, modes=None)
sliced_mix_data, window_parameters = Post_Process_Data.sliceTimePeriod(processed_mix_data, start=50, end=750)
sliced_new_real_data, _ = Post_Process_Data.sliceTimePeriod(processed_new_real_data, start=50, end=750)
# median filtering
filtered_mix_data = Post_Process_Data.spatialFilterModelInput(sliced_mix_data, kernel=3)
filtered_new_real_data = Post_Process_Data.spatialFilterModelInput(sliced_new_real_data, kernel=5)
del mix_old_emg_grids, new_real_emg_grids, processed_mix_data, processed_new_real_data, sliced_mix_data, sliced_new_real_data


## train classifier
mix_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
train_set, shuffled_train_set = mix_evaluation.classifierTrainSet(filtered_mix_data, dataset='cross_validation_set')
models_compare, model_results_compare = mix_evaluation.trainClassifier(shuffled_train_set)
acc_compare, cm_compare = mix_evaluation.evaluateClassifyResultsByGroup(model_results_compare)
# test classifier
test_set, shuffled_test_set = mix_evaluation.classifierTestSet(modes_generation, filtered_new_real_data, train_set, test_ratio=0.5)
test_results = mix_evaluation.testClassifier(models_compare, shuffled_test_set)
accuracy_compare, cm_recall_compare = mix_evaluation.evaluateClassifyResultsByGroup(test_results)


## plotting fake and real emg data for comparison
transition_type = 'emg_LWSA'
modes = modes_generation[transition_type]
# calculate average values
real_mix = Plot_Emg_Data.calcuAverageEmgValues(filtered_mix_data)
real_new = Plot_Emg_Data.calcuAverageEmgValues(filtered_new_real_data)
# plot values of certain transition type
Plot_Emg_Data.plotMultipleEventMeanValues(real_mix, real_new, modes, title='mix_emg_2', ylim=(0, 0.5))
# Plot_Emg_Data.plotMultipleRepetitionValues(real_mix, real_new, None, modes, ylim=(0, 1))
# Plot_Emg_Data.plotMultipleChannelValues(real_mix, real_new, None, modes, ylim=(0, 1))
# Plot_Emg_Data.plotMutipleEventPsdMeanValues(real_mix, real_new, None, modes, ylim=(0, 0.1), grid='grid_1')


## save results
model_type = 'classify_compare'
Model_Storage.saveClassifyResult(subject, accuracy_compare, cm_recall_compare, version, result_set, model_type, project='cGAN_Model')
accuracy_compare, cm_recall_compare = Model_Storage.loadClassifyResult(subject, version, result_set, model_type, project='cGAN_Model')
Model_Storage.saveClassifyModels(models_compare[0], subject, version, model_type, project='cGAN_Model')
model_compare = Model_Storage.loadClassifyModels(subject, version, model_type, project='cGAN_Model')




## load check point models
output = {}
# for transition_type in modes_generation.keys():
#     test_model = Model_Storage.loadCheckPointModels(checkpoint_model_path, model_name, epoch_number=30, transition_type=transition_type)
#     gen_model = cGAN_Testing.ModelTesting(test_model['gen'])
#     result = gen_model.estimateBlendingFactors(train_gan_data[transition_type], noise_dim=noise_dim)
#     output[transition_type] = result


