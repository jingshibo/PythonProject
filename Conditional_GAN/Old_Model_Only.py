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
classifier_filter_kernel = (20, 20)
plot_ylim = 1
basis_result_set = 0
filter_result_set = 0  # the results from different classifier filter kernel size are saved into different folders
reference = None  # value: None or num_reference, decide where to save the results

train_classifier = Train_Classifiers.ClassifierTraining(classifier_filter_kernel, gan_filter_kernel, window_parameters, modes_generation)
gen_results, old_emg_classify_normalized, new_emg_classify_normalized, extracted_emg_classify, window_parameters = \
    train_classifier.loadTrainingData(old_emg_data, new_emg_data, subject, version, gan_result_set, checkpoint_result_path, start_index,
        end_index, start_before_toeoff_ms, range_limit, time_interval, length, epoch_number)


'''
    train classifier (train on synthetic old data), for testing gan generation performance
'''
## build dataset and train classifier
models_old, accuracy_old, cm_recall_old, selected_old_fake_data, filtered_old_real_data = train_classifier.trainClassifierOldData(
    old_emg_classify_normalized, extracted_emg_classify, gen_results, num_sample=num_sample, num_ref=num_reference)
# plot data
train_classifier.plotEmgData(selected_old_fake_data['fake_data_based_on_grid_1'], filtered_old_real_data, plot_ylim=plot_ylim, title='')
## save model
# Model_Storage.saveClassifyResult(subject, accuracy_old, cm_recall_old, version, filter_result_set, 'classify_old', project='cGAN_Model', num_reference=reference)
# Model_Storage.saveClassifyModels(models_old, subject, version, 'classify_old', model_number=list(range(5)), project='cGAN_Model', num_reference=reference)


'''
    train classifier (train on real old + synthetic old data), for evaluating the proposed method performance
'''
## build dataset and train classifier
models_old_mix, accuracy_old_mix, cm_recall_old_mix, selected_old_mix_fake_data, adjusted_old_mix_real_data, \
            reference_old_mix_real_data, processed_old_mix_real_data, filtered_old_mix_real_data = train_classifier.trainClassifierOldMixData(
    old_emg_classify_normalized, extracted_emg_classify, gen_results, num_sample=num_sample, num_ref=num_reference)
# plot data
train_classifier.plotEmgData(selected_old_mix_fake_data['fake_data_based_on_grid_1'], adjusted_old_mix_real_data, plot_ylim=plot_ylim, title='')
## save model
# Model_Storage.saveClassifyResult(subject, accuracy_old_mix, cm_recall_old_mix, version, filter_result_set, 'classify_old_mix', project='cGAN_Model', num_reference=reference)
# Model_Storage.saveClassifyModels(models_old_mix, subject, version, 'classify_old_mix', model_number=list(range(5)), project='cGAN_Model', num_reference=reference)


'''
    train classifier (train on copying old data), select some reference new data without any other augmentation for training comparison
'''
## build dataset and train classifier
accuracy_old_copy, cm_recall_old_copy, models_old_copy, replicated_only_old_data = train_classifier.trainClassifierOldCopyData(
    filtered_old_mix_real_data, reference_old_mix_real_data, adjusted_old_mix_real_data, num_sample=num_sample)
## save results
# # Model_Storage.saveClassifyResult(subject, accuracy_old_copy, cm_recall_old_copy, version, filter_result_set, 'classify_old_copy', project='cGAN_Model', num_reference=reference)
# # Model_Storage.saveClassifyModels(models_old_copy, subject, version, 'classify_old_copy', model_number=list(range(5)), project='cGAN_Model', num_reference=reference)


## load check point models
output = {}
# for transition_type in modes_generation.keys():
#     test_model = Model_Storage.loadCheckPointModels(checkpoint_model_path, model_name, epoch_number=30, transition_type=transition_type)
#     gen_model = cGAN_Testing.ModelTesting(test_model['gen'])
#     result = gen_model.estimateBlendingFactors(train_gan_data[transition_type], noise_dim=noise_dim)
#     output[transition_type] = result

