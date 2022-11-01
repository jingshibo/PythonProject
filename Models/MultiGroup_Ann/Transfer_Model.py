'''
Pretrain a ann model using pre_train dataset first. Then train the pretrained model again using transfer_train dataset.
'''


## import modules
from Models.Basic_Ann.Functions import CV_Dataset, Ann_Model
from Models.MultiGroup_Ann.Functions import Multiple_Ann_Model, Grouped_CV_Dataset, Transfer_Learn_Dataset
import datetime
import os
import tensorflow as tf


## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use
fold = 5  # 5-fold cross validation

# read feature data
emg_features, emg_feature_reshaped = CV_Dataset.loadEmgFeature(subject, version, feature_set)
emg_feature_data = CV_Dataset.removeSomeMode(emg_features)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[0]  # how many windows there are for each event repetition
pre_train_dataset,  transfer_train_dataset = Transfer_Learn_Dataset.divideTransferDataset(fold, emg_feature_data)


## pre train model
# pre_train data
pretrain_normalized_groups = CV_Dataset.combineIntoDataset(pre_train_dataset, window_per_repetition)
pretrain_shuffled_groups = CV_Dataset.shuffleTrainingSet(pretrain_normalized_groups)

# classify using an basic ann model
now = datetime.datetime.now()
pretrain_model_results = Ann_Model.classifyUsingAnnModel(pretrain_shuffled_groups)
print(datetime.datetime.now() - now)
pretrain_majority_results = Ann_Model.majorityVoteResults(pretrain_model_results, window_per_repetition)
pretrain_average_accuracy, pretrain_sum_cm = Ann_Model.averageAccuracy(pretrain_majority_results)
pretrain_cm_recall = Ann_Model.confusionMatrix(pretrain_sum_cm, recall=True)
print(pretrain_cm_recall, '\n', pretrain_average_accuracy)


## transfer train model
# transfer_train data
transfer_transition_grouped = Grouped_CV_Dataset.separateGroups(transfer_train_dataset)
transfer_combined_groups = Grouped_CV_Dataset.combineIntoDataset(transfer_transition_grouped, window_per_repetition)
transfer_normalized_groups = Grouped_CV_Dataset.normalizeDataset(transfer_combined_groups)
transfer_shuffled_groups = Grouped_CV_Dataset.shuffleTrainingSet(transfer_normalized_groups)

# extract pretrained models
models = []
for model_result in pretrain_model_results:
    models.append(model_result['model'])

# classify using an transferred ann model
now = datetime.datetime.now()
transfer_model_results = Multiple_Ann_Model.classifyTransferAnnModel(transfer_shuffled_groups, models)
print(datetime.datetime.now() - now)
transfer_majority_results = Multiple_Ann_Model.majorityVoteResults(transfer_model_results, window_per_repetition)
transfer_accuracy, transfer_cm = Multiple_Ann_Model.getAccuracyPerGroup(transfer_majority_results)
transfer_average_accuracy, transfer_sum_cm = Multiple_Ann_Model.averageAccuracy(transfer_accuracy, transfer_cm)
transfer_cm_recall = Multiple_Ann_Model.confusionMatrix(transfer_sum_cm, recall=True)
print(transfer_average_accuracy, transfer_cm_recall)


## view trained models
# class_number = 4
# model = models[2]
# model.summary()
# pretrained_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
# x = tf.keras.layers.Dense(class_number, name='dense_new')(pretrained_model.layers[-1].output)
# output = tf.keras.layers.Softmax()(x)
# transferred_model = tf.keras.Model(inputs=model.input, outputs=output)
# for layer in transferred_model.layers[:-2]:
#     layer.trainable = False  # all pretrained layers are frozen
# transferred_model.summary()


# ## load trained model
# fold = 5
# type = 1  # pre-trained model type
# models = []
# for number in range(fold):
#     model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models_{type}\cross_validation_set_{number}'
#     models.append(tf.keras.models.load_model(model_dir))


# ## save trained models
# type = 1  # define the type of trained model to save
# for number, model in enumerate(model_results):
#     model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models_{type}\cross_validation_set_{number}'
#     if os.path.isfile(model_dir) is False:  # check if the location is a file or directory. Only save when it is a directory
#         model['model'].save(model_dir)  # save the model to the directory