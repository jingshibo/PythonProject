'''
    using conditional gan to generate transitional data and fine-train the classifier based on transfer learning. For each locomotion
    mode, only one prediction result is made, with no delay reported.
    This file is the sealed version of 'cGAN_TL_No_Sliding' for clarity. The function is exactly the same.
'''


##
from Conditional_GAN.Models import Model_Storage
from Conditional_GAN.Data_Procesing import Process_Raw_Data, Train_Classifiers, Train_cGan

'''train generative model'''
## read and filter old data
subject = 'Number1'
grid = 'grid_1'
version = 0  # the data from which experiment version to process
up_down_session_t0 = [0, 1, 2, 3, 4]
down_up_session_t0 = [1, 2, 3, 4, 5]
up_down_session_t1 = [0, 1, 2, 3, 4]
down_up_session_t1 = [1, 2, 3, 4, 5]
old_emg_data, new_emg_data, window_parameters, start_before_toeoff_ms = Train_cGan.realEmgData(subject, version, up_down_session_t0,
    down_up_session_t0, up_down_session_t1, down_up_session_t1, grid=grid)

## parameters for extracting emg data to train gan model
range_limit = 1500
gan_filter_kernel = (2, 1)
modes_generation = {'emg_LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'emg_LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD'],
    'emg_SALW': ['emg_SASA', 'emg_LWLW', 'emg_SALW'], 'emg_SDLW': ['emg_SDSD', 'emg_LWLW', 'emg_SDLW']} # The order in each list is important, corresponding to gen_data_1 and gen_data_2.
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']  # the length of data in each repetition
time_interval = 5  # the bin used to separate the time_series data in each repetition
old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped = Process_Raw_Data.normalizeFilterEmgData(old_emg_data,
    new_emg_data, range_limit, normalize='(0,1)', spatial_filter=True, kernel=gan_filter_kernel)
extracted_emg, train_gan_data = Process_Raw_Data.extractSeparateEmgData(modes_generation, old_emg_reshaped, new_emg_reshaped, time_interval,
    length, output_list=True)

## gan training hyperparameters
num_epochs = 30
decay_epochs = [30, 40]
batch_size = 1024  # maximum value to prevent overflow during running
sampling_repetition = 100  # the number of samples for each time point
gen_update_interval = 3  # The frequency at which the generator is updated. if set to 2, the generator is updated every 2 batches.
disc_update_interval = 1  # The frequency at which the discriminator is updated. if set to 2, the discriminator is updated every 2 batches.
noise_dim = 100
blending_factor_dim = 2
training_parameters = {'modes_generation': modes_generation, 'noise_dim': noise_dim, 'sampling_repetition': sampling_repetition,
    'gen_update_interval': gen_update_interval, 'disc_update_interval': disc_update_interval, 'batch_size': batch_size,
    'num_epochs': num_epochs, 'decay_epochs': decay_epochs, 'interval': time_interval, 'blending_factor_dim': blending_factor_dim}

## train and save gan models for multiple transitions
# GAN data storage information
checkpoint_model_path = f'D:\Data\cGAN_Model\subject_{subject}\Experiment_{version}\models\check_points'
checkpoint_result_path = f'D:\Data\cGAN_Model\subject_{subject}\Experiment_{version}\model_results\check_points'
model_type = 'cGAN'
model_name = ['gen', 'disc']
gan_result_set = 0
storage_parameters = {'subject': subject, 'version': version, 'model_type': model_type, 'model_name': model_name,
    'result_set': gan_result_set, 'checkpoint_model_path': checkpoint_model_path, 'checkpoint_result_path': checkpoint_result_path}
# train gan model
# blending_factors = Train_cGan.trainCGan(train_gan_data, modes_generation, training_parameters, storage_parameters)
del old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped, extracted_emg, train_gan_data

## plotting fake and real emg data for comparison
# Train_cGan.plotEmgData(old_emg_data, new_emg_data, ylim=(0, 1500))



'''
    load data for classifier training
'''
## laod training data
# load blending factors for each transition type to generate
epoch_number = None
start_index = 400  # start index relative to the start of the original extracted data
end_index = 1600  # end index relative to the start of the original extracted data
num_sample = 50
num_reference = 1
classifier_filter_kernel = (40, 40)
plot_ylim = 0.2
classifier_result_set = 0
reference = None  # value: None or num_reference, decide where to save the results

train_classifier = Train_Classifiers.ClassifierTraining(classifier_filter_kernel, gan_filter_kernel, window_parameters, modes_generation)
gen_results, old_emg_classify_normalized, new_emg_classify_normalized, extracted_emg_classify, window_parameters = \
    train_classifier.loadTrainingData(old_emg_data, new_emg_data, subject, version, gan_result_set, checkpoint_result_path, start_index,
        end_index, start_before_toeoff_ms, range_limit, time_interval, length, epoch_number)


'''
    train classifier (basic scenarios), training and testing using data from the same and different time
'''
## train classifier (basic scenarios), training and testing data from the same and different time
models_basis, accuracy_basis, cm_recall_basis, accuracy_best, cm_recall_best, accuracy_worst, cm_recall_worst, accuracy_tf, cm_recall_tf = \
    train_classifier.trainClassifierBasicScenarios(old_emg_classify_normalized, new_emg_classify_normalized)
## save models
# Model_Storage.saveClassifyResult(subject, accuracy_basis, cm_recall_basis, version, classifier_result_set, 'classify_basis', project='cGAN_Model')
# Model_Storage.saveClassifyResult(subject, accuracy_best, cm_recall_best, version, classifier_result_set, 'classify_best', project='cGAN_Model')
# Model_Storage.saveClassifyResult(subject, accuracy_tf, cm_recall_tf, version, classifier_result_set, 'classify_tf', project='cGAN_Model')
# Model_Storage.saveClassifyResult(subject, accuracy_worst, cm_recall_worst, version, classifier_result_set, 'classify_worst', project='cGAN_Model')
# Model_Storage.saveClassifyModels(models_basis, subject, version, 'classify_basis', model_number=list(range(5)), project='cGAN_Model')


'''
    train classifier (on old data), for testing gan generation performance
'''
## build dataset and train classifier
models_old, accuracy_old, cm_recall_old, selected_old_fake_data, filtered_old_real_data = train_classifier.trainClassifierOldData(
    old_emg_classify_normalized, extracted_emg_classify, gen_results, num_sample=num_sample, num_ref=num_reference)
# plot data
train_classifier.plotEmgData(selected_old_fake_data['fake_data_based_on_grid_1'], filtered_old_real_data, plot_ylim=plot_ylim, title='old')
## save model
# Model_Storage.saveClassifyResult(subject, accuracy_old, cm_recall_old, version, classifier_result_set, 'classify_old', project='cGAN_Model', num_reference=reference)
# Model_Storage.saveClassifyModels(models_old, subject, version, 'classify_old', model_number=list(range(5)), project='cGAN_Model', num_reference=reference)


'''
    train classifier (on new data), for evaluating the proposed method performance
'''
## build dataset and train classifier
models_new, accuracy_new, cm_recall_new, selected_new_fake_data, adjusted_new_real_data, reference_new_real_data, \
    processed_new_real_data, filtered_new_real_data = train_classifier.trainClassifierNewData(
    new_emg_classify_normalized, extracted_emg_classify, gen_results, models_basis, num_sample=num_sample, num_ref=num_reference)
# plot data
train_classifier.plotEmgData(selected_new_fake_data['fake_data_based_on_grid_1'], adjusted_new_real_data, plot_ylim=plot_ylim, title='new')
## save model
# Model_Storage.saveClassifyResult(subject, accuracy_new, cm_recall_new, version, classifier_result_set, 'classify_new', project='cGAN_Model', num_reference=reference)
# Model_Storage.saveClassifyModels(models_new, subject, version, 'classify_new', model_number=list(range(5)), project='cGAN_Model', num_reference=reference)


'''
    train classifier (on old and new data), for comparison purpose
'''
## build dataset and train classifier
accuracy_compare, cm_recall_compare, models_compare, filtered_mix_data = train_classifier.trainClassifierMixData(old_emg_classify_normalized,
    new_emg_classify_normalized, reference_new_real_data, adjusted_new_real_data, models_basis)
# plot data
train_classifier.plotEmgData(filtered_mix_data, adjusted_new_real_data, plot_ylim=plot_ylim, title='mix')
## save results
# Model_Storage.saveClassifyResult(subject, accuracy_compare, cm_recall_compare, version, classifier_result_set, 'classify_compare', project='cGAN_Model', num_reference=reference)
# Model_Storage.saveClassifyModels(models_compare, subject, version, 'classify_compare', model_number=list(range(5)), project='cGAN_Model', num_reference=reference)


'''
    train classifier (on old and fake new data), for improvement purpose
'''
## build dataset and train classifier
accuracy_combine, cm_recall_combine, models_combine, filtered_combined_data = train_classifier.trainClassifierCombineData(
    selected_new_fake_data, filtered_mix_data, reference_new_real_data, adjusted_new_real_data, models_basis)
# plot data
train_classifier.plotEmgData(filtered_combined_data, adjusted_new_real_data, plot_ylim=plot_ylim, title='combine')
## save results
# Model_Storage.saveClassifyResult(subject, accuracy_combine, cm_recall_combine, version, classifier_result_set, 'classify_combine', project='cGAN_Model', num_reference=reference)
# Model_Storage.saveClassifyModels(models_combine, subject, version, 'classify_combine', model_number=list(range(5)), project='cGAN_Model', num_reference=reference)


'''
    train classifier (on noisy new data), select some reference new data and augment them with noise for training comparison
'''
## build dataset and train classifier
accuracy_noise, cm_recall_noise, models_noise, filtered_noise_data = train_classifier.trainClassifierNoiseData(processed_new_real_data,
    reference_new_real_data, adjusted_new_real_data, models_basis, num_sample=num_sample)
## save results
# Model_Storage.saveClassifyResult(subject, accuracy_noise, cm_recall_noise, version, classifier_result_set, 'classify_noise', project='cGAN_Model', num_reference=reference)
# Model_Storage.saveClassifyModels(models_noise, subject, version, 'classify_noise', model_number=list(range(5)), project='cGAN_Model', num_reference=reference)


'''
    train classifier (on copying new data), select some reference new data without any other augmentation for training comparison
'''
## build dataset and train classifier
accuracy_copy, cm_recall_copy, models_copy, replicated_only_new_data = train_classifier.trainClassifierCopyData(filtered_new_real_data,
    reference_new_real_data, adjusted_new_real_data, models_basis, num_sample=num_sample)
## save results
# Model_Storage.saveClassifyResult(subject, accuracy_copy, cm_recall_copy, version, classifier_result_set, 'classify_copy', project='cGAN_Model', num_reference=reference)
# Model_Storage.saveClassifyModels(models_copy, subject, version, 'classify_copy', model_number=list(range(5)), project='cGAN_Model', num_reference=reference)


## load check point models
output = {}
# for transition_type in modes_generation.keys():
#     test_model = Model_Storage.loadCheckPointModels(checkpoint_model_path, model_name, epoch_number=30, transition_type=transition_type)
#     gen_model = cGAN_Testing.ModelTesting(test_model['gen'])
#     result = gen_model.estimateBlendingFactors(train_gan_data[transition_type], noise_dim=noise_dim)
#     output[transition_type] = result