##
from Conditional_GAN.Models import cGAN_Training, cGAN_Testing, cGAN_Evaluation, Model_Storage
from Conditional_GAN.Data_Procesing import Process_Fake_Data, Process_Raw_Data
import datetime


'''train generative model'''
##  define windows
window_parameters = Process_Raw_Data.ganWindowParameters()

## read and filter old data
subject = 'Number4'
version = 0  # the data from which experiment version to process
modes_generation = ['up_down', 'down_up']
up_down_session = [0, 1, 2, 3, 4]
down_up_session = [0, 1, 2, 5, 6]
# up_down_session = [0]
# down_up_session = [0]
sessions = [up_down_session, down_up_session]
data_source = {'subject': subject, 'version': version, 'modes': modes_generation, 'sessions': sessions}
old_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=20, higher_limit=400)


## read and filter new data
subject = 'Number5'
version = 0  # the data from which experiment version to process
modes_generation = ['up_down', 'down_up']
up_down_session = [5, 6, 7, 8, 9]
down_up_session = [4, 5, 6, 8, 9]
# up_down_session = [0]
# down_up_session = [0]
sessions = [up_down_session, down_up_session]
data_source = {'subject': subject, 'version': version, 'modes': modes_generation, 'sessions': sessions}
new_emg_data = Process_Raw_Data.readFilterEmgData(data_source, window_parameters, lower_limit=20, higher_limit=400)


## normalize and extract emg data for gan model training
range_limit = 1500
old_emg_normalized, new_emg_normalized, old_emg_reshaped, new_emg_reshaped = Process_Raw_Data.normalizeReshapeEmgData(old_emg_data, new_emg_data, range_limit)
modes_generation = ['emg_LWLW', 'emg_SASA', 'emg_LWSA']
length = window_parameters['start_before_toeoff_ms'] + window_parameters['endtime_after_toeoff_ms']
time_interval = 50
extracted_emg, train_gan_data = Process_Raw_Data.extractSeparateEmgData(modes_generation, old_emg_reshaped, new_emg_reshaped, time_interval, length, output_list=True)


## hyperparameters
num_epochs = 200  # the number of times you iterate through the entire dataset when training
decay_epochs = 10
batch_size = 1024  # the number of images per forward/backward pass
sampling_repetition = 5  # the number of batches to repeat the combination sampling for the same time points
noise_dim = 5
blending_factor_dim = 2


## GAN data storage information
subject = 'Test'
version = 0  # the data from which experiment version to process
checkpoint_model_path = f'D:\Data\cGAN_Model\subject_{subject}\Experiment_{version}\models\check_points'
checkpoint_result_path = f'D:\Data\cGAN_Model\subject_{subject}\Experiment_{version}\model_results\check_points'
model_type = 'cGAN'
model_name = ['gen', 'disc']
result_set = 0


## train gan model
now = datetime.datetime.now()
train_model = cGAN_Training.ModelTraining(num_epochs, batch_size, sampling_repetition, decay_epochs, noise_dim, blending_factor_dim)
gan_models, blending_factors = train_model.trainModel(train_gan_data, checkpoint_model_path, checkpoint_result_path)
print(datetime.datetime.now() - now)


## save trained gan models and results
Model_Storage.saveModels(gan_models, subject, version, model_type, model_name, project='cGAN_Model')
# save model results
training_parameters = {'noise_dim': noise_dim, 'sampling_repetition': sampling_repetition, 'batch_size': batch_size,
    'num_epochs': num_epochs, 'interval': time_interval, 'blending_factor_dim': blending_factor_dim}
Model_Storage.saveCGanResults(subject, blending_factors, version, result_set, training_parameters, model_type, project='cGAN_Model')





'''
    train classifier (on subject 4), for testing gan generation performance
'''
## load blending factors
gen_result = Model_Storage.loadCGanResults(subject, version, result_set, model_type, project='cGAN_Model')
checkpoint_result = Model_Storage.loadCheckPointCGanResults(checkpoint_result_path, epoch_number=200)
gen_result['model_results'] = checkpoint_result
gen_results = {'LWSA': gen_result}  # there could be more than one transition to generate


## generate fake data
old_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
modes_generation = {'LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA']}  # The order in the list is critical
synthetic_data = old_evaluation.generateFakeData(extracted_emg, 'old', modes_generation, old_emg_normalized, repetition=1, random_pairing=False)

## train classifier
train_set, shuffled_train_set = old_evaluation.classifierTrainSet(synthetic_data, training_percent=0.8)
models, model_results = old_evaluation.trainClassifier(shuffled_train_set)
# accuracy, cm_recall = old_evaluation.evaluateClassifyResults(model_results)

## test classifier
shuffled_test_set = old_evaluation.classifierTestSet(modes_generation, old_emg_normalized, train_set, test_ratio=0.5)
test_results = old_evaluation.testClassifier(models[0], shuffled_test_set)
accuracy_old, cm_recall_old = old_evaluation.evaluateClassifyResults(test_results)

## save results
model_type = 'classify_old'
Model_Storage.saveClassifyResult(subject, accuracy_old, cm_recall_old, version, result_set, model_type, project='cGAN_Model')
acc, cm = Model_Storage.loadClassifyResult(subject, version, result_set, model_type, project='cGAN_Model')
Model_Storage.saveClassifyModel(models[0], subject, version, model_type, project='cGAN_Model')
model = Model_Storage.loadClassifyModel(subject, version, model_type, project='cGAN_Model')



'''
    train classifier (on subject 5), for evaluating the proposed method performance
'''
## generate fake data
new_evaluation = cGAN_Evaluation.cGAN_Evaluation(gen_results, window_parameters)
modes_generation = {'LWSA': ['emg_LWLW', 'emg_SASA', 'emg_LWSA']}  # The order in the list is critical
synthetic_data = new_evaluation.generateFakeData(extracted_emg, 'new', modes_generation, new_emg_normalized, repetition=1, random_pairing=False)

## train classifier
train_set, shuffled_train_set = new_evaluation.classifierTrainSet(synthetic_data, training_percent=0.8)
models, model_results = new_evaluation.trainClassifier(shuffled_train_set)
# accuracy, cm_recall = new_evaluation.evaluateClassifyResults(model_results)

## test classifier
shuffled_test_set = new_evaluation.classifierTestSet(modes_generation, new_emg_normalized, train_set, test_ratio=0.5)
test_results = new_evaluation.testClassifier(models[0], shuffled_test_set)
accuracy_new, cm_recall_new = new_evaluation.evaluateClassifyResults(test_results)

## save results
model_type = 'classify_new'
Model_Storage.saveClassifyResult(subject, accuracy_new, cm_recall_new, version, result_set, model_type, project='cGAN_Model')
acc, cm = Model_Storage.loadClassifyResult(subject, version, result_set, model_type, project='cGAN_Model')
Model_Storage.saveClassifyModel(models[0], subject, version, model_type, project='cGAN_Model')
model = Model_Storage.loadClassifyModel(subject, version, model_type, project='cGAN_Model')



'''
    train classifier (on subject 5), for comparison purpose
'''
## build training data
old_emg_data = {'emg_LWSA': old_emg_normalized['emg_LWSA']}
synthetic_data = Process_Fake_Data.replaceUsingFakeEmg(old_emg_data, new_emg_normalized)

## train classifier
train_set, shuffled_train_set = new_evaluation.classifierTrainSet(synthetic_data, training_percent=0.8)
models, model_results = new_evaluation.trainClassifier(shuffled_train_set)
# accuracy, cm_recall = new_evaluation.evaluateClassifyResults(model_results)

## test classifier
shuffled_test_set = new_evaluation.classifierTestSet(modes_generation, new_emg_normalized, train_set, test_ratio=0.5)
test_results = new_evaluation.testClassifier(models[0], shuffled_test_set)
accuracy_compare, cm_recall_compare = new_evaluation.evaluateClassifyResults(test_results)

## save results
model_type = 'classify_compare'
Model_Storage.saveClassifyResult(subject, accuracy_compare, cm_recall_compare, version, result_set, model_type, project='cGAN_Model')
acc, cm = Model_Storage.loadClassifyResult(subject, version, result_set, model_type, project='cGAN_Model')
Model_Storage.saveClassifyModel(models[0], subject, version, model_type, project='cGAN_Model')
model = Model_Storage.loadClassifyModel(subject, version, model_type, project='cGAN_Model')






## load check point models
test_model = Model_Storage.loadCheckPointModels(checkpoint_model_path, model_name, epoch_number=50)
gen_model = cGAN_Testing.ModelTesting(test_model['gen'])
results = gen_model.estimateBlendingFactors(train_gan_data, noise_dim=noise_dim)



