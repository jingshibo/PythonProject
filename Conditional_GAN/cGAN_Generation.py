##
from Conditional_GAN.Models import cGAN_Training, cGAN_Testing, cGAN_Evaluation, Model_Storage
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data
import datetime


'''train generative model'''
##  define windows
window_parameters = Process_Raw_Data.returnWindowParameters()
lower_limit = 20
higher_limit = 400
envelope_cutoff = 10
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
old_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    envelope_cutoff=envelope_cutoff, envelope=envelope)
# old_emg_data_classify = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
#     envelope_cutoff=400, envelope=envelope)


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
new_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
    envelope_cutoff=envelope_cutoff, envelope=envelope)
# new_emg_data_classify = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=lower_limit, higher_limit=higher_limit,
#     envelope_cutoff=400, envelope=envelope)


## test stable values
import numpy as np
# keys = {'emg_LWLW': 200, 'emg_LWSA': 400, 'emg_SASA': 600, 'emg_LWSD': 150, 'emg_SDSD': 100}
# for key, value in keys.items():
#     for i, array in enumerate(old_emg_data[key]):
#         noise = 10 * np.random.randn(*array.shape)
#         old_emg_data[key][i] = np.full_like(array, value) + noise


## test changing values
import numpy as np

# increasing_values = np.linspace(100, 1000, 900)
# reshaped_values = increasing_values[:, np.newaxis]
# emg_LWLW = np.tile(reshaped_values, (1, 130))
# decreasing_values = np.linspace(1000, 100, 900)
# reshaped_values = decreasing_values[:, np.newaxis]
# emg_SASA = np.tile(reshaped_values, (1, 130))
#
# increasing_values = np.linspace(100, 1000, 450)
# decreasing_values = np.linspace(1000, 100, 450)
# combined_values = np.concatenate((increasing_values, decreasing_values))
# reshaped_values = combined_values[:, np.newaxis]
# emg_LWSA = np.tile(reshaped_values, (1, 130))


## constant values
array_first_half = np.full((450, 130), 100)
array_second_half = np.full((450, 130), 900)
emg_LWLW = np.vstack([array_first_half, array_second_half])

array_first_half = np.full((450, 130), 500)
array_second_half = np.full((450, 130), 500)
emg_SASA = np.vstack([array_first_half, array_second_half])

emg_LWSA = np.full((900, 130), 200)

# Define the keys dictionary
keys = {'emg_LWLW': emg_LWLW, 'emg_SASA': emg_SASA, 'emg_LWSA': emg_LWSA}

# Loop through each key-value pair in the keys dictionary
for key, value in keys.items():
    for i, array in enumerate(old_emg_data[key]):
        noise = 5 * np.random.randn(*array.shape)
        old_emg_data[key][i] = value + noise


## normalize and extract emg data for gan model training
range_limit = 2000
old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped = Process_Raw_Data.normalizeReshapeEmgData(old_emg_data,
    new_emg_data, range_limit, normalize='(0,1)')
# The order in each list is important, corresponding to gen_data_1 and gen_data_2.
# modes_generation = {'LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA'], 'LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD']}
modes_generation = {'LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA']}
# modes_generation = {'LWSD': ['emg_LWLW', 'emg_SDSD', 'emg_LWSD']}
time_interval = 25
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']
extracted_emg, train_gan_data = Process_Raw_Data.extractSeparateEmgData(modes_generation, old_emg_reshaped, new_emg_reshaped, time_interval,
    length, output_list=True)


## hyperparameters
num_epochs = 100
decay_epochs = [30, 45]
batch_size = 1024  # this is also the number of samples to extract for each time_interval
sampling_repetition = 1  # the number of batches to repeat the extraction of samples for the same time points
gen_update_interval = 3  # The frequency at which the generator is updated. if set to 2, the generator is updated every 2 batches.
disc_update_interval = 1  # The frequency at which the discriminator is updated. if set to 2, the discriminator is updated every 2 batches.
noise_dim = 0
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
epoch_number = 10
gen_results = Model_Storage.loadBlendingFactors(subject, version, result_set, model_type, modes_generation, checkpoint_result_path, epoch_number=epoch_number)
old_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
synthetic_data = old_evaluation.generateFakeData(extracted_emg, 'old', modes_generation, old_emg_normalized, repetition=1, random_pairing=False)


'''
    train classifier (on subject 4), for testing gan generation performance
'''
## laod training data
# load blending factors for each transition type to generate
epoch_number = None
gen_results = Model_Storage.loadBlendingFactors(subject, version, result_set, model_type, modes_generation, checkpoint_result_path, epoch_number=epoch_number)
# normalize and extract emg data for classification model training
old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped = Process_Raw_Data.normalizeReshapeEmgData(old_emg_data_classify,
    new_emg_data_classify, range_limit, normalize='(0,1)')
extracted_emg, _ = Process_Raw_Data.extractSeparateEmgData(modes_generation, old_emg_reshaped, new_emg_reshaped, time_interval,
    length, output_list=True)


## generate fake data
old_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
synthetic_data = old_evaluation.generateFakeData(extracted_emg, 'old', modes_generation, old_emg_normalized, repetition=1, random_pairing=False)
# train classifier
train_set, shuffled_train_set = old_evaluation.classifierTrainSet(synthetic_data, training_percent=0.8)
models_old, model_result_old = old_evaluation.trainClassifier(shuffled_train_set)
acc_old, cm_old = old_evaluation.evaluateClassifyResults(model_result_old)
# test classifier
shuffled_test_set = old_evaluation.classifierTestSet(modes_generation, old_emg_normalized, train_set, test_ratio=0.5)
test_results = old_evaluation.testClassifier(models_old[0], shuffled_test_set)
accuracy_old, cm_recall_old = old_evaluation.evaluateClassifyResults(test_results)


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
synthetic_data = new_evaluation.generateFakeData(extracted_emg, 'new', modes_generation, new_emg_normalized, repetition=1, random_pairing=False)
# train classifier
train_set, shuffled_train_set = new_evaluation.classifierTrainSet(synthetic_data, training_percent=0.8)
models_new, model_results_new = new_evaluation.trainClassifier(shuffled_train_set)
acc_new, cm_new = new_evaluation.evaluateClassifyResults(model_results_new)
# test classifier
shuffled_test_set = new_evaluation.classifierTestSet(modes_generation, new_emg_normalized, train_set, test_ratio=0.5)
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
old_emg_for_replacement = {modes[2]: old_emg_normalized[modes[2]] for transition_type, modes in modes_generation.items()}
synthetic_data = Process_Fake_Data.replaceUsingFakeEmg(old_emg_for_replacement, new_emg_normalized)
# train classifier
train_set, shuffled_train_set = new_evaluation.classifierTrainSet(synthetic_data, training_percent=0.8)
models_compare, model_results_compare = new_evaluation.trainClassifier(shuffled_train_set)
acc_compare, cm_compare = new_evaluation.evaluateClassifyResults(model_results_compare)
# test classifier
shuffled_test_set = new_evaluation.classifierTestSet(modes_generation, new_emg_normalized, train_set, test_ratio=0.5)
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
    test_model = Model_Storage.loadCheckPointModels(checkpoint_model_path, model_name, epoch_number=50, transition_type=transition_type)
    gen_model = cGAN_Testing.ModelTesting(test_model['gen'])
    result = gen_model.estimateBlendingFactors(train_gan_data[transition_type], noise_dim=0)
    output[transition_type] = result


