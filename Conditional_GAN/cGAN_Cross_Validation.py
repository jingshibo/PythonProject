'''
    using conditional gan to generate transitional data and fine-train the classifier based on transfer learning. For each locomotion
    mode, only one prediction result is made, with no delay reported. This version introduces cross validation to the gan training data set,
    splitting the dataset into training set and test set
'''


##
import copy
import datetime
from Conditional_GAN.Models import cGAN_Training, cGAN_Testing, cGAN_Evaluation, Model_Storage
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data, Plot_Emg_Data, Dtw_Similarity, Post_Process_Data
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset
from Transition_Prediction.Models.Utility_Functions import Data_Preparation


'''train generative model'''
##  define windows
start_before_toeoff_ms = 1000
endtime_after_toeoff_ms = 1000
feature_window_ms = start_before_toeoff_ms + endtime_after_toeoff_ms
predict_window_ms = start_before_toeoff_ms + endtime_after_toeoff_ms
window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=start_before_toeoff_ms,
    endtime_after_toeoff_ms=endtime_after_toeoff_ms, feature_window_ms=feature_window_ms, predict_window_ms=predict_window_ms)
lower_limit = 20
higher_limit = 400
envelope_cutoff = 400
envelope = True  # the output will always be rectified if set True


## read and filter old data
subject = 'Number8'
version = 0  # the data from which experiment version to process
modes = ['up_down_t0', 'down_up_t0']
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [0, 1, 2, 5, 6]
up_down_session = [5, 6, 7, 8, 9]
down_up_session = [7, 8, 9, 10]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [1, 2, 3, 4, 5]
# up_down_session = [0, 1, 2, 3, 4, 5]
# down_up_session = [0, 1, 2, 3, 4, 5]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [0, 1, 2, 3, 4]
# up_down_session = [5, 6, 7, 8, 9, 10]
# down_up_session = [6, 8, 9, 10]
# up_down_session = [0, 1, 2, 3, 4, 5]
# down_up_session = [0, 1, 2, 3, 4, 5]
sessions = [up_down_session, down_up_session]
data_source = {'subject': subject, 'version': version, 'modes': modes, 'sessions': sessions}
# old_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
#     envelope_cutoff=envelope_cutoff, envelope=envelope)
old_emg_data_classify = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    envelope_cutoff=envelope_cutoff, envelope=envelope, project='cGAN_Model', selected_grid='grid_2', include_standing=False)


# read and filter new data
modes = ['up_down_t1', 'down_up_t1']
# up_down_session = [5, 6, 7, 8, 9]
# down_up_session = [4, 5, 6, 8, 9]
up_down_session = [0, 1, 2, 3, 4]
down_up_session = [0, 1, 2, 3]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [1, 2, 3, 4, 5]
# up_down_session = [0, 1, 2, 3, 4, 5]
# down_up_session = [1, 2, 3, 4, 5]
# up_down_session = [0, 1, 2, 3]
# down_up_session = [0, 1, 2, 3]
# up_down_session = [5, 6, 7, 8, 9, 10]
# down_up_session = [6, 7, 8, 9, 10]
# up_down_session = [0, 1, 2, 3, 4, 5]
# down_up_session = [0, 1, 3, 4]
sessions = [up_down_session, down_up_session]
data_source = {'subject': subject, 'version': version, 'modes': modes, 'sessions': sessions}
# new_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
#     envelope_cutoff=envelope_cutoff, envelope=envelope)
new_emg_data_classify = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    envelope_cutoff=envelope_cutoff, envelope=envelope, project='cGAN_Model', selected_grid='grid_2', include_standing=False)


##  gan model training setup
range_limit = 1500
gan_filter_kernel = (2, 1)
# The order in each list is important, corresponding to gen_data_1 and gen_data_2.
modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'],
    'emg_SALW': ['emg_SASA', 'emg_LWLW', 'emg_SALW'], 'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}
# modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD']}
# modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA']}
# modes_generation = {'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}
time_interval = 5
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']


## normalize and extract emg data for gan model training
old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped = Process_Raw_Data.normalizeFilterEmgData(old_emg_data_classify,
    new_emg_data_classify, range_limit, normalize='(0,1)', spatial_filter=True, kernel=gan_filter_kernel)
gan_old_training_set, gan_new_training_set, _, _ = Process_Raw_Data.splitGanDataset(old_emg_reshaped,
    new_emg_reshaped, training_percent=0.8)  # split data for training set and test set
extracted_emg, train_gan_data = Process_Raw_Data.extractSeparateEmgData(modes_generation, gan_old_training_set, gan_new_training_set,
    time_interval, length, output_list=True)


## hyperparameters
num_epochs = 30
decay_epochs = [30, 40]
batch_size = 1024  # maximum value to prevent overflow during running
sampling_repetition = 100  # the number of samples for each time point
gen_update_interval = 3  # The frequency at which the generator is updated. if set to 2, the generator is updated every 2 batches.
disc_update_interval = 1  # The frequency at which the discriminator is updated. if set to 2, the discriminator is updated every 2 batches.
noise_dim = 100
blending_factor_dim = 2


# GAN data storage information
checkpoint_model_path = f'D:\Data\cGAN_Model\subject_{subject}\Experiment_{version}\models\check_points'
checkpoint_result_path = f'D:\Data\cGAN_Model\subject_{subject}\Experiment_{version}\model_results\check_points'
model_type = 'cGAN'
model_name = ['gen', 'disc']
gan_result_set = 3


## train and save gan models for multiple transitions
training_parameters = {'modes_generation': modes_generation, 'noise_dim': noise_dim, 'sampling_repetition': sampling_repetition,
    'gen_update_interval': gen_update_interval, 'disc_update_interval': disc_update_interval, 'batch_size': batch_size,
    'num_epochs': num_epochs, 'decay_epochs': decay_epochs, 'interval': time_interval, 'blending_factor_dim': blending_factor_dim}
storage_parameters = {'subject': subject, 'version': version, 'model_type': model_type, 'model_name': model_name,
    'result_set': gan_result_set, 'checkpoint_model_path': checkpoint_model_path, 'checkpoint_result_path': checkpoint_result_path}
now = datetime.datetime.now()
results = {}
for transition_type in modes_generation.keys():
    gan_models, blending_factors = cGAN_Training.trainCGan(train_gan_data[transition_type], transition_type, training_parameters, storage_parameters)
    results[transition_type] = blending_factors
print(datetime.datetime.now() - now)
del old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped, gan_old_training_set, gan_new_training_set, extracted_emg, train_gan_data


# ## plotting fake and real emg data for comparison
# # calculate average values
# real_old = Plot_Emg_Data.calcuAverageEmgValues(old_emg_data_classify)
# real_new = Plot_Emg_Data.calcuAverageEmgValues(new_emg_data_classify)
# # plot multiple repetition values in a subplot
# ylim=(0, 1500)
# Plot_Emg_Data.plotAverageValue(real_old['emg_repetition_list']['grid_1'], 'emg_SASA', num_columns=30, layout=None, title='old_SASA', ylim=ylim)
# Plot_Emg_Data.plotAverageValue(real_new['emg_repetition_list']['grid_1'], 'emg_SASA', num_columns=30, layout=None, title='new_SASA', ylim=ylim)
# Plot_Emg_Data.plotAverageValue(real_old['emg_repetition_list']['grid_1'], 'emg_LWSA', num_columns=30, layout=None, title='old_LWSA', ylim=ylim)
# Plot_Emg_Data.plotAverageValue(real_new['emg_repetition_list']['grid_1'], 'emg_LWSA', num_columns=30, layout=None, title='new_LWSA', ylim=ylim)
# Plot_Emg_Data.plotAverageValue(real_old['emg_repetition_list']['grid_1'], 'emg_LWLW', num_columns=30, layout=None, title='old_LWLW', ylim=ylim)
# Plot_Emg_Data.plotAverageValue(real_new['emg_repetition_list']['grid_1'], 'emg_LWLW', num_columns=30, layout=None, title='new_LWLW', ylim=ylim)
# Plot_Emg_Data.plotAverageValue(real_old['emg_repetition_list']['grid_1'], 'emg_SDSD', num_columns=30, layout=None, title='old_SDSD', ylim=ylim)
# Plot_Emg_Data.plotAverageValue(real_new['emg_repetition_list']['grid_1'], 'emg_SDSD', num_columns=30, layout=None, title='new_SDSD', ylim=ylim)




'''
    load data for classifier training
'''
## laod training data
# load blending factors for each transition type to generate
epoch_number = None
model_type = 'cGAN'
gen_results = Model_Storage.loadBlendingFactors(subject, version, gan_result_set, model_type, modes_generation, checkpoint_result_path,
    epoch_number=epoch_number)
# gen_results = Process_Raw_Data.spatialFilterBlendingFactors(gen_result, kernel=gan_filter_kernel) # filter blending factor
# normalize and extract emg data for classification model training
old_emg_classify_normalized, new_emg_classify_normalized, old_emg_classify_reshaped, new_emg_classify_reshaped = \
    Process_Raw_Data.normalizeFilterEmgData(old_emg_data_classify, new_emg_data_classify, range_limit, normalize='(0,1)',
        spatial_filter=True, kernel=gan_filter_kernel)
_, _, gan_old_test_set, gan_new_test_set = Process_Raw_Data.splitGanDataset(old_emg_classify_reshaped,
    new_emg_classify_reshaped, training_percent=0.8)  # split data for training set and test set
extracted_emg_classify, _ = Process_Raw_Data.extractSeparateEmgData(modes_generation, gan_old_test_set, gan_new_test_set,
    time_interval, length, output_list=False)
# classifier training parameters
# start_index = 50  # start index relative to the start of the original extracted data
# end_index = 750  # end index relative to the start of the original extracted data
start_index = 0
end_index = 2000
classifier_filter_kernel = (10, 10)
classifier_result_set = 3



'''
    train classifier (basic scenarios), training and testing data from the same and different time
'''
## original dataset
old_real_emg_grids = Post_Process_Data.separateEmgGrids(old_emg_classify_normalized, separate=True)
new_real_emg_grids = Post_Process_Data.separateEmgGrids(new_emg_classify_normalized, separate=True)
processed_old_real_data = Process_Fake_Data.reorderSmoothDataSet(old_real_emg_grids['grid_1'], filtering=False, modes=None)
processed_new_real_data = Process_Fake_Data.reorderSmoothDataSet(new_real_emg_grids['grid_1'], filtering=False, modes=None)
sliced_old_real_data, window_parameters = Post_Process_Data.sliceTimePeriod(processed_old_real_data, start=start_index, end=end_index,
    toeoff=start_before_toeoff_ms, sliding_results=False)
sliced_new_real_data, _ = Post_Process_Data.sliceTimePeriod(processed_new_real_data, start=start_index, end=end_index,
    toeoff=start_before_toeoff_ms, sliding_results=False)
filtered_old_real_data = Post_Process_Data.spatialFilterModelInput(sliced_old_real_data, kernel=classifier_filter_kernel)
filtered_new_real_data = Post_Process_Data.spatialFilterModelInput(sliced_new_real_data, kernel=classifier_filter_kernel)
del old_real_emg_grids, new_real_emg_grids, processed_old_real_data, processed_new_real_data

## classification
# use old data to train old model
basis_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
old_train_set, shuffled_train_set = basis_evaluation.classifierTrainSet(filtered_old_real_data, dataset='cross_validation_set')
models_basis, model_result_basis = basis_evaluation.trainClassifier(shuffled_train_set, num_epochs=50, batch_size=64, decay_epochs=20)
accuracy_basis, cm_recall_basis = basis_evaluation.evaluateClassifyResults(model_result_basis)
# use new data to train new model
best_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
new_train_set, shuffled_train_set = best_evaluation.classifierTrainSet(filtered_new_real_data, dataset='cross_validation_set')
models_best, model_result_best = best_evaluation.trainClassifier(shuffled_train_set, num_epochs=50, batch_size=64, decay_epochs=20)
accuracy_best, cm_recall_best = best_evaluation.evaluateClassifyResults(model_result_best)  # training and testing data from the same time
# using old model to classify new data
cross_validation_groups = Data_Preparation.crossValidationSet(5, filtered_new_real_data)
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(cross_validation_groups,
    window_parameters['feature_window_size'], increment=window_parameters['feature_window_increment_ms'] * window_parameters['sample_rate'])
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset, normalize=None)
shuffled_test_set = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
test_results = basis_evaluation.testClassifier(models_basis, shuffled_test_set)
accuracy_worst, cm_recall_worst = basis_evaluation.evaluateClassifyResults(test_results)  # training and testing data from different time
del old_train_set, new_train_set, shuffled_train_set, cross_validation_groups, sliding_window_dataset, normalized_groups, shuffled_test_set

## save results
model_type = 'classify_basis'
Model_Storage.saveClassifyResult(subject, accuracy_basis, cm_recall_basis, version, classifier_result_set, model_type, project='cGAN_Model')
accuracy_basis, cm_recall_basis = Model_Storage.loadClassifyResult(subject, version, classifier_result_set, model_type, project='cGAN_Model')
model_type = 'classify_best'
Model_Storage.saveClassifyResult(subject, accuracy_best, cm_recall_best, version, classifier_result_set, model_type, project='cGAN_Model')
accuracy_best, cm_recall_best = Model_Storage.loadClassifyResult(subject, version, classifier_result_set, model_type, project='cGAN_Model')
model_type = 'classify_worst'
Model_Storage.saveClassifyResult(subject, accuracy_worst, cm_recall_worst, version, classifier_result_set, model_type, project='cGAN_Model')
accuracy_worst, cm_recall_worst = Model_Storage.loadClassifyResult(subject, version, classifier_result_set, model_type, project='cGAN_Model')
model_type = 'classify_basis'
Model_Storage.saveClassifyModels(models_basis, subject, version, model_type, model_number=list(range(5)), project='cGAN_Model')
models_basis = Model_Storage.loadClassifyModels(subject, version, model_type, model_number=list(range(5)), project='cGAN_Model')



'''
    train classifier (on old data), for testing gan generation performance
'''
## generate fake data
window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=start_before_toeoff_ms,
    endtime_after_toeoff_ms=endtime_after_toeoff_ms, feature_window_ms=feature_window_ms, predict_window_ms=predict_window_ms)
old_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
synthetic_old_data, fake_old_images = old_evaluation.generateFakeData(extracted_emg_classify, 'old', modes_generation,
    old_emg_classify_normalized, spatial_filtering=False, kernel=gan_filter_kernel)
# separate and store grids in a list if only use one grid later
old_fake_emg_grids = Post_Process_Data.separateEmgGrids(synthetic_old_data, separate=True)
old_real_emg_grids = Post_Process_Data.separateEmgGrids(old_emg_classify_normalized, separate=True)
del synthetic_old_data

## only preprocess selected grid and define time range of data
processed_old_fake_data = Process_Fake_Data.reorderSmoothDataSet(old_fake_emg_grids['grid_1'], filtering=False, modes=modes_generation)
processed_old_real_data = Process_Fake_Data.reorderSmoothDataSet(old_real_emg_grids['grid_1'], filtering=False, modes=None)
sliced_old_fake_data, window_parameters = Post_Process_Data.sliceTimePeriod(processed_old_fake_data, start=start_index, end=end_index,
    toeoff=start_before_toeoff_ms, sliding_results=False)
sliced_old_real_data, _ = Post_Process_Data.sliceTimePeriod(processed_old_real_data, start=start_index, end=end_index,
    toeoff=start_before_toeoff_ms, sliding_results=False)
del old_fake_emg_grids, old_real_emg_grids, processed_old_fake_data, processed_old_real_data

## select representative fake data for classification model training
filtered_old_fake_data = Post_Process_Data.spatialFilterModelInput(sliced_old_fake_data, kernel=classifier_filter_kernel)
filtered_old_real_data = Post_Process_Data.spatialFilterModelInput(sliced_old_real_data, kernel=classifier_filter_kernel)
# select representative fake data for classification model training
selected_old_fake_data = Dtw_Similarity.extractFakeData(filtered_old_fake_data, filtered_old_real_data, modes_generation,
    envelope_frequency=None, num_sample=100, num_reference=1, method='select', random_reference=False, split_grids=True)

## classification
old_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
train_set, shuffled_train_set = old_evaluation.classifierTrainSet(selected_old_fake_data['fake_data_based_on_grid_1'], dataset='cross_validation_set')
models_old, model_result_old = old_evaluation.trainClassifier(shuffled_train_set, num_epochs=50, batch_size=64, decay_epochs=20)
acc_old, cm_old = old_evaluation.evaluateClassifyResults(model_result_old)
# test classifier
test_set, shuffled_test_set = old_evaluation.classifierTestSet(modes_generation, filtered_old_real_data, train_set, test_ratio=1)
test_results = old_evaluation.testClassifier(models_old, shuffled_test_set)
accuracy_old, cm_recall_old = old_evaluation.evaluateClassifyResults(test_results)
del train_set, shuffled_train_set, test_set, shuffled_test_set

## plotting fake and real emg data for comparison
transition_type = 'emg_LWSD'
modes = modes_generation[transition_type]
# calculate average values
# reference_old_data = {transition_type: [sliced_old_real_data[transition_type][index] for index in
#     selected_old_fake_data['reference_index_based_on_grid_1'][transition_type]]}
fake_old = Plot_Emg_Data.calcuAverageEmgValues(selected_old_fake_data['fake_data_based_on_grid_1'])
real_old = Plot_Emg_Data.calcuAverageEmgValues(filtered_old_real_data)
# reference_old = Plot_Emg_Data.averageEmgValues(reference_old_data)
# plot values of certain transition type
Plot_Emg_Data.plotMultipleEventMeanValues(fake_old, real_old, modes, title='old_emg_2_on_2', ylim=(0, 0.5))
transition_type = 'emg_SDLW'
modes = modes_generation[transition_type]
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
# Plot_Emg_Data.plotMainEventMeanValues(fake_old, real_old, title='old_main_emg', ylim=(0, 0.5), grid='grid_1')

## save results
model_type = 'classify_old'
Model_Storage.saveClassifyResult(subject, accuracy_old, cm_recall_old, version, classifier_result_set, model_type, project='cGAN_Model')
accuracy_old, cm_recall_old = Model_Storage.loadClassifyResult(subject, version, classifier_result_set, model_type, project='cGAN_Model')
Model_Storage.saveClassifyModels(models_old, subject, version, model_type, model_number=list(range(5)), project='cGAN_Model')
models_old = Model_Storage.loadClassifyModels(subject, version, model_type, model_number=list(range(5)), project='cGAN_Model')



'''
    train classifier (on new data), for evaluating the proposed method performance
'''
## generate fake data
window_parameters = Process_Raw_Data.returnWindowParameters(start_before_toeoff_ms=start_before_toeoff_ms,
    endtime_after_toeoff_ms=endtime_after_toeoff_ms, feature_window_ms=feature_window_ms, predict_window_ms=predict_window_ms)
new_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
synthetic_new_data, fake_new_images = new_evaluation.generateFakeData(extracted_emg_classify, 'new', modes_generation,
    new_emg_classify_normalized, spatial_filtering=False, kernel=gan_filter_kernel)
# separate and store grids in a list if only use one grid later
new_fake_emg_grids = Post_Process_Data.separateEmgGrids(synthetic_new_data, separate=True)
new_real_emg_grids = Post_Process_Data.separateEmgGrids(new_emg_classify_normalized, separate=True)
del synthetic_new_data

# only preprocess selected grid and define time range of data
processed_new_fake_data = Process_Fake_Data.reorderSmoothDataSet(new_fake_emg_grids['grid_1'], filtering=False, modes=modes_generation)
processed_new_real_data = Process_Fake_Data.reorderSmoothDataSet(new_real_emg_grids['grid_1'], filtering=False, modes=None)
sliced_new_fake_data, window_parameters = Post_Process_Data.sliceTimePeriod(processed_new_fake_data, start=start_index, end=end_index,
    toeoff=start_before_toeoff_ms, sliding_results=False)
sliced_new_real_data, _ = Post_Process_Data.sliceTimePeriod(processed_new_real_data, start=start_index, end=end_index,
    toeoff=start_before_toeoff_ms, sliding_results=False)
del new_fake_emg_grids, new_real_emg_grids, processed_new_fake_data, processed_new_real_data

## post-process dataset for model input
filtered_new_fake_data = Post_Process_Data.spatialFilterModelInput(sliced_new_fake_data, kernel=classifier_filter_kernel)
filtered_new_real_data = Post_Process_Data.spatialFilterModelInput(sliced_new_real_data, kernel=classifier_filter_kernel)
# select representative fake data for classification model training
selected_new_fake_data = Dtw_Similarity.extractFakeData(filtered_new_fake_data, filtered_new_real_data, modes_generation,
    envelope_frequency=None, num_sample=100, num_reference=1, method='select', random_reference=False, split_grids=True)

## classification
new_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
reference_indices = selected_new_fake_data['reference_index_based_on_grid_1']
reference_new_real_data, adjusted_new_real_data = new_evaluation.addressReferenceData(reference_indices, filtered_new_real_data)
train_set, shuffled_train_set = new_evaluation.classifierTlTrainSet(selected_new_fake_data['fake_data_based_on_grid_1'],
    reference_new_real_data, dataset='cross_validation_set', minimum_train_number=10)
models_new, model_results_new = new_evaluation.trainTlClassifier(models_basis, shuffled_train_set, num_epochs=50, batch_size=32, decay_epochs=20)
acc_new, cm_new = new_evaluation.evaluateClassifyResults(model_results_new)
# test classifier
test_set, shuffled_test_set = new_evaluation.classifierTestSet(modes_generation, adjusted_new_real_data, train_set, test_ratio=1)
test_results = new_evaluation.testClassifier(models_new, shuffled_test_set)
accuracy_new, cm_recall_new = new_evaluation.evaluateClassifyResults(test_results)
del train_set, shuffled_train_set, test_set, shuffled_test_set

## plotting fake and real emg data for comparison
# calculate average values
fake_new = Plot_Emg_Data.calcuAverageEmgValues(selected_new_fake_data['fake_data_based_on_grid_1'])
real_new = Plot_Emg_Data.calcuAverageEmgValues(filtered_new_real_data)
# reference_new = Plot_Emg_Data.calcuAverageEmgValues(reference_new_real_data)
# plot values of certain transition type
transition_type = 'emg_LWSD'
modes = modes_generation[transition_type]
Plot_Emg_Data.plotMultipleEventMeanValues(fake_new, real_new, modes, title='new_emg_2_on_2', ylim=(0, 0.5))
transition_type = 'emg_SDLW'
modes = modes_generation[transition_type]
Plot_Emg_Data.plotMultipleEventMeanValues(fake_new, real_new, modes, title='new_emg_2_on_2', ylim=(0, 0.5))
# Plot_Emg_Data.plotMultipleRepetitionValues(fake_new, real_new, reference_new, modes, ylim=(0, 1))
# Plot_Emg_Data.plotMultipleChannelValues(fake_new, real_new, reference_new, modes, ylim=(0, 1))
# Plot_Emg_Data.plotMutipleEventPsdMeanValues(fake_new, real_new, reference_new, modes, ylim=(0, 0.1), grid='grid_1')
# plot the dtw distance curve (between a selected fake data and reference data)
# new_fake_envelopes = selected_new_fake_data['fake_envelope_averaged']['emg_repetition_list']['grid_1']['emg_LWSA']
# new_real_envelopes = selected_new_fake_data['real_envelope_averaged']['emg_repetition_list']['grid_1']['emg_LWSA']
# new_fake_envelope_index = selected_new_fake_data['fake_index_based_on_grid_1']['emg_LWSA'][1]
# new_real_envelope_index = selected_new_fake_data['reference_index_based_on_grid_1']['emg_LWSA'][1]
# Dtw_Similarity.plotDtwPath(new_fake_envelopes, new_real_envelopes, new_fake_envelope_index, new_real_envelope_index)
Plot_Emg_Data.plotMainEventMeanValues(fake_new, real_new, title='new_main_emg', ylim=(0, 0.5), grid='grid_1')

## save results
model_type = 'classify_new'
Model_Storage.saveClassifyResult(subject, accuracy_new, cm_recall_new, version, classifier_result_set, model_type, project='cGAN_Model')
accuracy_new, cm_recall_new = Model_Storage.loadClassifyResult(subject, version, classifier_result_set, model_type, project='cGAN_Model')
Model_Storage.saveClassifyModels(models_new, subject, version, model_type, model_number=list(range(5)), project='cGAN_Model')
models_new = Model_Storage.loadClassifyModels(subject, version, model_type, model_number=list(range(5)), project='cGAN_Model')



'''
    train classifier (on old and new data), for comparison purpose
'''
## build training data
old_emg_for_replacement = {modes[2]: old_emg_classify_normalized[modes[2]] for transition_type, modes in modes_generation.items()}
mix_old_new_data = Process_Fake_Data.replaceUsingFakeEmg(old_emg_for_replacement, new_emg_classify_normalized)
# separate and store grids in a list if only use one grid later
mix_old_new_grids = Post_Process_Data.separateEmgGrids(mix_old_new_data, separate=True)

## only preprocess selected grid and define time range of data
processed_mix_data = Process_Fake_Data.reorderSmoothDataSet(mix_old_new_grids['grid_1'], filtering=False, modes=modes_generation)
sliced_mix_data, window_parameters = Post_Process_Data.sliceTimePeriod(processed_mix_data, start=start_index, end=end_index,
    toeoff=start_before_toeoff_ms, sliding_results=False)
# median filtering
filtered_mix_data = Post_Process_Data.spatialFilterModelInput(sliced_mix_data, kernel=classifier_filter_kernel)
del mix_old_new_data, mix_old_new_grids, processed_mix_data, sliced_mix_data

## classification
mix_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
# use the same reference new data above as the available new data for the mix dataset, to adjust both the train_set and test_set
train_set, shuffled_train_set = mix_evaluation.classifierTlTrainSet(filtered_mix_data, reference_new_real_data, dataset='cross_validation_set')
models_compare, model_results_compare = mix_evaluation.trainTlClassifier(models_basis, shuffled_train_set, num_epochs=50, batch_size=32, decay_epochs=20)
acc_compare, cm_compare = mix_evaluation.evaluateClassifyResults(model_results_compare)
# test classifier
test_set, shuffled_test_set = mix_evaluation.classifierTestSet(modes_generation, adjusted_new_real_data, train_set, test_ratio=1)
test_results = mix_evaluation.testClassifier(models_compare, shuffled_test_set)
accuracy_compare, cm_recall_compare = mix_evaluation.evaluateClassifyResults(test_results)
del train_set, shuffled_train_set, test_set, shuffled_test_set

## plotting fake and real emg data for comparison
transition_type = 'emg_LWSD'
modes = modes_generation[transition_type]
# calculate average values
real_mix = Plot_Emg_Data.calcuAverageEmgValues(filtered_mix_data)
real_new = Plot_Emg_Data.calcuAverageEmgValues(filtered_new_real_data)
# plot values of certain transition type
Plot_Emg_Data.plotMultipleEventMeanValues(real_mix, real_new, modes, title='mix_emg_2', ylim=(0, 0.5))
transition_type = 'emg_SDLW'
modes = modes_generation[transition_type]
Plot_Emg_Data.plotMultipleEventMeanValues(real_mix, real_new, modes, title='mix_emg_2', ylim=(0, 0.5))
# Plot_Emg_Data.plotMultipleRepetitionValues(real_mix, real_new, None, modes, ylim=(0, 1))
# Plot_Emg_Data.plotMultipleChannelValues(real_mix, real_new, None, modes, ylim=(0, 1))
# Plot_Emg_Data.plotMutipleEventPsdMeanValues(real_mix, real_new, None, modes, ylim=(0, 0.1), grid='grid_1')

## save results
model_type = 'classify_compare'
Model_Storage.saveClassifyResult(subject, accuracy_compare, cm_recall_compare, version, classifier_result_set, model_type, project='cGAN_Model')
accuracy_compare, cm_recall_compare = Model_Storage.loadClassifyResult(subject, version, classifier_result_set, model_type, project='cGAN_Model')
Model_Storage.saveClassifyModels(models_compare, subject, version, model_type, model_number=list(range(5)), project='cGAN_Model')
models_compare = Model_Storage.loadClassifyModels(subject, version, model_type, model_number=list(range(5)), project='cGAN_Model')



'''
    train classifier (on noisy new data), select some reference new data and augment them with noise for training comparison
'''
## generate noise data
noise_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)  # window_parameters are in line with the above
noise_new_data = noise_evaluation.generateNoiseData(sliced_new_real_data, reference_new_real_data, num_sample=100, snr=0.05)
# median filtering
filtered_noise_data = Post_Process_Data.spatialFilterModelInput(noise_new_data, kernel=classifier_filter_kernel)

## classification
noise_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
# use the same reference new data above as the available new data for the mix dataset, to adjust both the train_set and test_set
train_set, shuffled_train_set = noise_evaluation.classifierTlTrainSet(filtered_noise_data, reference_new_real_data, dataset='cross_validation_set')
models_noise, model_results_noise = noise_evaluation.trainTlClassifier(models_basis, shuffled_train_set, num_epochs=50, batch_size=32, decay_epochs=20)
acc_noise, cm_noise = noise_evaluation.evaluateClassifyResults(model_results_noise)
# test classifier
test_set, shuffled_test_set = noise_evaluation.classifierTestSet(modes_generation, adjusted_new_real_data, train_set, test_ratio=1)
test_results = noise_evaluation.testClassifier(models_noise, shuffled_test_set)
accuracy_noise, cm_recall_noise = noise_evaluation.evaluateClassifyResults(test_results)

## save results
model_type = 'classify_noise'
Model_Storage.saveClassifyResult(subject, accuracy_noise, cm_recall_noise, version, classifier_result_set, model_type, project='cGAN_Model')
accuracy_noise, cm_recall_noise = Model_Storage.loadClassifyResult(subject, version, classifier_result_set, model_type, project='cGAN_Model')
Model_Storage.saveClassifyModels(models_noise, subject, version, model_type, model_number=list(range(5)), project='cGAN_Model')
models_noise = Model_Storage.loadClassifyModels(subject, version, model_type, model_number=list(range(5)), project='cGAN_Model')


## load check point models
output = {}
# for transition_type in modes_generation.keys():
#     test_model = Model_Storage.loadCheckPointModels(checkpoint_model_path, model_name, epoch_number=30, transition_type=transition_type)
#     gen_model = cGAN_Testing.ModelTesting(test_model['gen'])
#     result = gen_model.estimateBlendingFactors(train_gan_data[transition_type], noise_dim=noise_dim)
#     output[transition_type] = result


