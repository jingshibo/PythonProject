'''
    using conditional gan to generate transitional data and fine-train the classifier based on transfer learning. For each locomotion
    mode, only one prediction result is made, with no delay reported.
'''


##
import random
from Conditional_GAN.Models import cGAN_Evaluation, Model_Storage, Train_Classifiers
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data, Plot_Emg_Data, Dtw_Similarity, Post_Process_Data

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
subject = 'Number0'
version = 0  # the data from which experiment version to process
modes = ['up_down_t0', 'down_up_t0']
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [0, 1, 2, 5, 6]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [1, 2, 3, 4, 5]
# up_down_session = [5, 6, 7, 8, 9]
# down_up_session = [6, 8, 9, 10]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [0, 1, 2, 3, 4]
# up_down_session = [0, 1, 2, 4, 5]
# down_up_session = [0, 1, 2, 3, 4]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [0, 1, 2, 3, 4]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [2, 3, 4, 5]
up_down_session = [0, 1, 2, 3, 4]
down_up_session = [0, 1, 2, 3, 4]
# up_down_session = [5, 6, 7, 8, 9]
# down_up_session = [7, 8, 9, 10]
sessions = [up_down_session, down_up_session]
data_source = {'subject': subject, 'version': version, 'modes': modes, 'sessions': sessions}
# old_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
#     envelope_cutoff=envelope_cutoff, envelope=envelope)
old_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    envelope_cutoff=envelope_cutoff, envelope=envelope, project='cGAN_Model', selected_grid='grid_1', include_standing=False)


# read and filter new data
modes = ['up_down_t1', 'down_up_t1']
# up_down_session = [5, 6, 7, 8, 9]
# down_up_session = [4, 5, 6, 8, 9]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [1, 2, 3, 4, 5]
# up_down_session = [5, 6, 7, 8, 9]
# down_up_session = [6, 7, 8, 9, 10]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [1, 2, 3, 4, 5]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [0, 1, 3, 4]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [0, 1, 2, 3, 4]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [0, 1, 2, 3, 4]
up_down_session = [0, 1, 2, 3]
down_up_session = [0, 1, 2, 3]
# up_down_session = [0, 1, 2, 3, 4]
# down_up_session = [0, 1, 2, 3]
sessions = [up_down_session, down_up_session]
data_source = {'subject': subject, 'version': version, 'modes': modes, 'sessions': sessions}
# new_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
#     envelope_cutoff=envelope_cutoff, envelope=envelope)
new_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    envelope_cutoff=envelope_cutoff, envelope=envelope, project='cGAN_Model', selected_grid='grid_1', include_standing=False)


## normalize and extract emg data for gan model training
range_limit = 1500
gan_filter_kernel = (2, 1)
old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped = Process_Raw_Data.normalizeFilterEmgData(old_emg_data,
    new_emg_data, range_limit, normalize='(0,1)', spatial_filter=True, kernel=gan_filter_kernel)
# The order in each list is important, corresponding to gen_data_1 and gen_data_2.
modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'],
    'emg_SALW': ['emg_SASA', 'emg_LWLW', 'emg_SALW'], 'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']}
time_interval = 5
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']
extracted_emg, train_gan_data = Process_Raw_Data.extractSeparateEmgData(modes_generation, old_emg_reshaped, new_emg_reshaped, time_interval,
    length, output_list=True)


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
gan_result_set = 0


## train and save gan models for multiple transitions
training_parameters = {'modes_generation': modes_generation, 'noise_dim': noise_dim, 'sampling_repetition': sampling_repetition,
    'gen_update_interval': gen_update_interval, 'disc_update_interval': disc_update_interval, 'batch_size': batch_size,
    'num_epochs': num_epochs, 'decay_epochs': decay_epochs, 'interval': time_interval, 'blending_factor_dim': blending_factor_dim}
storage_parameters = {'subject': subject, 'version': version, 'model_type': model_type, 'model_name': model_name,
    'result_set': gan_result_set, 'checkpoint_model_path': checkpoint_model_path, 'checkpoint_result_path': checkpoint_result_path}
# now = datetime.datetime.now()
# results = {}
# for transition_type in modes_generation.keys():
#     gan_models, blending_factors = cGAN_Training.trainCGan(train_gan_data[transition_type], transition_type, training_parameters, storage_parameters)
#     results[transition_type] = blending_factors
# print(datetime.datetime.now() - now)
del old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped, extracted_emg, train_gan_data


## plotting fake and real emg data for comparison
# # calculate average values
# real_old = Plot_Emg_Data.calcuAverageEmgValues(old_emg_data)
# real_new = Plot_Emg_Data.calcuAverageEmgValues(new_emg_data)
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
start_index = 400  # start index relative to the start of the original extracted data
end_index = 1600  # end index relative to the start of the original extracted data
classifier_filter_kernel = (40, 40)
plot_ylim = 0.3
classifier_result_set = 0

train_classifier = Train_Classifiers.ClassifierTraining(classifier_filter_kernel, gan_filter_kernel, window_parameters, modes_generation)
gen_results, old_emg_classify_normalized, new_emg_classify_normalized, extracted_emg_classify = train_classifier.loadTrainingData(
    old_emg_data, new_emg_data, subject, version, gan_result_set, checkpoint_result_path, start_index, end_index, start_before_toeoff_ms,
    range_limit, time_interval, length, epoch_number)



'''
    train classifier (basic scenarios), training and testing using data from the same and different time
'''
## train classifier (basic scenarios), training and testing data from the same and different time
models_basis, accuracy_basis, cm_recall_basis, accuracy_best, cm_recall_best, accuracy_worst, cm_recall_worst = \
    train_classifier.trainClassifierBasicScenarios(old_emg_classify_normalized, new_emg_classify_normalized)
## save models
Model_Storage.saveClassifyResult(subject, accuracy_basis, cm_recall_basis, version, classifier_result_set, 'classify_basis', project='cGAN_Model')
Model_Storage.saveClassifyResult(subject, accuracy_best, cm_recall_best, version, classifier_result_set, 'classify_best', project='cGAN_Model')
Model_Storage.saveClassifyResult(subject, accuracy_worst, cm_recall_worst, version, classifier_result_set, 'classify_worst', project='cGAN_Model')
Model_Storage.saveClassifyModels(models_basis, subject, version, 'classify_basis', model_number=list(range(5)), project='cGAN_Model')



'''
    train classifier (on old data), for testing gan generation performance
'''
## train classifier (on old data), for testing gan generation performance
models_old, accuracy_old, cm_recall_old, selected_old_fake_data, filtered_old_real_data = train_classifier.trainClassifierOldData(
    old_emg_classify_normalized, extracted_emg_classify, gen_results)
## save model
Model_Storage.saveClassifyResult(subject, accuracy_old, cm_recall_old, version, classifier_result_set, 'classify_old', project='cGAN_Model')
Model_Storage.saveClassifyModels(models_old, subject, version, 'classify_old', model_number=list(range(5)), project='cGAN_Model')
## plot data
train_classifier.plotEmgData(selected_old_fake_data['fake_data_based_on_grid_1'], filtered_old_real_data, plot_ylim=plot_ylim, title='old')



'''
    train classifier (on new data), for evaluating the proposed method performance
'''
## generate fake data
models_new, accuracy_new, cm_recall_new, selected_new_fake_data, adjusted_new_real_data, reference_new_real_data, processed_new_real_data = \
    train_classifier.trainClassifierNewData(new_emg_classify_normalized, extracted_emg_classify, gen_results, models_basis)
## save model
Model_Storage.saveClassifyResult(subject, accuracy_new, cm_recall_new, version, classifier_result_set, 'classify_new', project='cGAN_Model')
Model_Storage.saveClassifyModels(models_new, subject, version, 'classify_new', model_number=list(range(5)), project='cGAN_Model')
## plot data
train_classifier.plotEmgData(selected_new_fake_data['fake_data_based_on_grid_1'], adjusted_new_real_data, plot_ylim=plot_ylim, title='new')



'''
    train classifier (on old and new data), for comparison purpose
'''
##
accuracy_compare, cm_recall_compare, models_compare, filtered_mix_data = train_classifier.trainClassifierMixData(old_emg_classify_normalized,
    new_emg_classify_normalized, reference_new_real_data, adjusted_new_real_data, models_basis)
## save results
Model_Storage.saveClassifyResult(subject, accuracy_compare, cm_recall_compare, version, classifier_result_set, 'classify_compare', project='cGAN_Model')
Model_Storage.saveClassifyModels(models_compare, subject, version, 'classify_compare', model_number=list(range(5)), project='cGAN_Model')
## plot data
train_classifier.plotEmgData(filtered_mix_data, adjusted_new_real_data, plot_ylim=plot_ylim, title='mix')



'''
    train classifier (on old and fake new data), for improvement purpose
'''
## build training data
accuracy_combine, cm_recall_combine, models_combine, filtered_combined_data = train_classifier.trainClassifierCombineData(
    selected_new_fake_data, filtered_mix_data, reference_new_real_data, adjusted_new_real_data, models_basis)
## save results
Model_Storage.saveClassifyResult(subject, accuracy_combine, cm_recall_combine, version, classifier_result_set, 'classify_combine', project='cGAN_Model')
Model_Storage.saveClassifyModels(models_combine, subject, version, 'classify_combine', model_number=list(range(5)), project='cGAN_Model')
## plot data
train_classifier.plotEmgData(filtered_combined_data, adjusted_new_real_data, plot_ylim=plot_ylim, title='combine')



'''
    train classifier (on noisy new data), select some reference new data and augment them with noise for training comparison
'''
## generate noise data
accuracy_noise, cm_recall_noise, models_noise, filtered_noise_data = train_classifier.trainClassifierNoiseData(processed_new_real_data,
    reference_new_real_data, adjusted_new_real_data, models_basis)
## save results
Model_Storage.saveClassifyResult(subject, accuracy_noise, cm_recall_noise, version, classifier_result_set, 'classify_noise', project='cGAN_Model')
Model_Storage.saveClassifyModels(models_noise, subject, version, 'classify_noise', model_number=list(range(5)), project='cGAN_Model')


## load check point models
output = {}
# for transition_type in modes_generation.keys():
#     test_model = Model_Storage.loadCheckPointModels(checkpoint_model_path, model_name, epoch_number=30, transition_type=transition_type)
#     gen_model = cGAN_Testing.ModelTesting(test_model['gen'])
#     result = gen_model.estimateBlendingFactors(train_gan_data[transition_type], noise_dim=noise_dim)
#     output[transition_type] = result



'''
    train classifier (average old and fake new data), for improvement purpose
'''
## build training data
# data before spatial filtering, random selection
# sampled_new_fake_data = cGAN_Evaluation.selectRandomData(processed_new_fake_data, modes_generation, num_samples=100)
# average_fake_data = cGAN_Evaluation.calculateAverageValues(sampled_new_fake_data, processed_mix_data, modes_generation)
# filtered_average_fake_data = Post_Process_Data.spatialFilterModelInput(average_fake_data, kernel=classifier_filter_kernel)
# filtered_new_fake_selected = Dtw_Similarity.extractFakeData(filtered_average_fake_data, filtered_new_real_data, modes_generation,
#     envelope_frequency=None, num_sample=100, num_reference=1, method='random', random_reference=False, split_grids=True)

# data before spatial filtering, dtw selection
# sampled_new_fake_data = cGAN_Evaluation.selectRandomData(processed_new_fake_data, modes_generation, num_samples=100)
# average_fake_data = cGAN_Evaluation.calculateAverageValues(sampled_new_fake_data, processed_mix_data, modes_generation)
# filtered_average_fake_data = Post_Process_Data.spatialFilterModelInput(average_fake_data, kernel=classifier_filter_kernel)
# filtered_new_fake_selected = Dtw_Similarity.extractFakeData(filtered_average_fake_data, filtered_new_real_data, modes_generation,
#     envelope_frequency=None, num_sample=100, num_reference=1, method='select', random_reference=False, split_grids=True)
#
# # data after spatial filtering, random selection
# sampled_new_fake_data = cGAN_Evaluation.selectRandomData(filtered_new_fake_data, modes_generation, num_samples=100)
# average_fake_data = cGAN_Evaluation.calculateAverageValues(sampled_new_fake_data, filtered_mix_data, modes_generation)
# filtered_new_fake_selected = Dtw_Similarity.extractFakeData(average_fake_data, filtered_new_real_data, modes_generation,
#     envelope_frequency=None, num_sample=100, num_reference=1, method='random', random_reference=False, split_grids=True)
#
# # data after spatial filtering, dtw selection
# average_fake_data = cGAN_Evaluation.calculateAverageValues(selected_new_fake_data['fake_data_based_on_grid_1'], filtered_mix_data, modes_generation)
# filtered_new_fake_selected = Dtw_Similarity.extractFakeData(average_fake_data, filtered_new_real_data, modes_generation,
#     envelope_frequency=None, num_sample=100, num_reference=1, method='select', random_reference=False, split_grids=True)
#
# ## classification
# mean_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
# # use the same reference new data above as the available new data for the mix dataset, to adjust both the train_set and test_set
# train_set, shuffled_train_set = mean_evaluation.classifierTlTrainSet(filtered_new_fake_selected['fake_data_based_on_grid_1'],
#     reference_new_real_data, dataset='cross_validation_set')
# models_average, model_results_average = mean_evaluation.trainTlClassifier(models_basis, shuffled_train_set, num_epochs=30, batch_size=32, decay_epochs=10)
# acc_average, cm_average = mean_evaluation.evaluateClassifyResults(model_results_average)
# # test classifier
# test_set, shuffled_test_set = mean_evaluation.classifierTestSet(modes_generation, adjusted_new_real_data, train_set, test_ratio=1)
# test_results = mean_evaluation.testClassifier(models_average, shuffled_test_set)
# accuracy_average, cm_recall_average = mean_evaluation.evaluateClassifyResults(test_results)
# del train_set, shuffled_train_set, test_set, shuffled_test_set
#
# ## plotting fake and real emg data for comparison
# # calculate average values
# real_average = Plot_Emg_Data.calcuAverageEmgValues(filtered_new_fake_selected['fake_data_based_on_grid_1'])
# real_new = Plot_Emg_Data.calcuAverageEmgValues(adjusted_new_real_data)
# # plot values of certain transition type
# transition_type = 'emg_LWSA'
# modes = modes_generation[transition_type]
# Plot_Emg_Data.plotMultipleEventMeanValues(real_average, real_new, modes, title='average_emg_2', ylim=(0, plot_y_limit))
# transition_type = 'emg_SALW'
# modes = modes_generation[transition_type]
# Plot_Emg_Data.plotMultipleEventMeanValues(real_average, real_new, modes, title='average_emg_2', ylim=(0, plot_y_limit))
# transition_type = 'emg_LWSD'
# modes = modes_generation[transition_type]
# Plot_Emg_Data.plotMultipleEventMeanValues(real_average, real_new, modes, title='average_emg_2', ylim=(0, plot_y_limit))
# transition_type = 'emg_SDLW'
# modes = modes_generation[transition_type]
# Plot_Emg_Data.plotMultipleEventMeanValues(real_average, real_new, modes, title='average_emg_2', ylim=(0, plot_y_limit))
# # Plot_Emg_Data.plotMultipleRepetitionValues(real_mix, real_new, None, modes, ylim=(0, 1))
# # Plot_Emg_Data.plotMultipleChannelValues(real_mix, real_new, None, modes, ylim=(0, 1))
# # Plot_Emg_Data.plotMutipleEventPsdMeanValues(real_mix, real_new, None, modes, ylim=(0, 0.1), grid='grid_1')