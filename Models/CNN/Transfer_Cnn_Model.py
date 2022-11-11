'''
Pretrain a cnn model using the pre_train dataset first. Then train the pretrained model by group using the transfer_train dataset.
'''


## import modules
from Models.Utility_Functions import Data_Preparation, MV_Results, MV_Results_ByGroup
from Models.CNN.Functions import Cnn_Dataset, Cnn_Model, Grouped_Cnn_Dataset
import datetime


## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use


# read feature data
emg_features, emg_feature_2d = Data_Preparation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Preparation.removeSomeMode(emg_feature_2d)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[-1]  # how many windows there are for each event repetition
fold = 5  # 5-fold cross validation
transfer_data_percent = 0.5  # percentage of dataset specifically for transfer learning divided from training set
pretrain_dataset,  transfer_train_dataset = Data_Preparation.divideTransferDataset(fold, emg_feature_data, transfer_data_percent)


## pre train model
# pre_train data
pretrain_normalized_groups = Cnn_Dataset.combineNormalizedDataset(pretrain_dataset, window_per_repetition)
pretrain_shuffled_groups = Cnn_Dataset.shuffleTrainingSet(pretrain_normalized_groups)

# classify using an basic ann model
now = datetime.datetime.now()
pretrain_model_results = Cnn_Model.classifyUsingCnnModel(pretrain_shuffled_groups)
print(datetime.datetime.now() - now)

# majority vote results
pretrain_majority_results = MV_Results.majorityVoteResults(pretrain_model_results, window_per_repetition)
pretrain_average_accuracy, pretrain_sum_cm = MV_Results.averageAccuracy(pretrain_majority_results)
pretrain_cm_recall = MV_Results.confusionMatrix(pretrain_sum_cm, recall=True)
print(pretrain_cm_recall, '\n', pretrain_average_accuracy)


## transfer train model
# transfer_train data
transfer_transition_grouped = Data_Preparation.separateGroups(transfer_train_dataset)
transfer_combined_groups = Grouped_Cnn_Dataset.combineIntoDataset(transfer_transition_grouped, window_per_repetition)
transfer_normalized_groups = Grouped_Cnn_Dataset.normalizeDataset(transfer_combined_groups)
transfer_shuffled_groups = Grouped_Cnn_Dataset.shuffleTrainingSet(transfer_normalized_groups)

# read pretrained models
models = []
for model_result in pretrain_model_results:
    models.append(model_result['model'])

# classify using an transferred ann model
now = datetime.datetime.now()
transfer_model_results = Cnn_Model.classifyTransferCnnModel(transfer_shuffled_groups, models)
print(datetime.datetime.now() - now)

# majority vote results
transfer_majority_results = MV_Results_ByGroup.majorityVoteResults(transfer_model_results, window_per_repetition)
transfer_accuracy, transfer_cm = MV_Results_ByGroup.getAccuracyPerGroup(transfer_majority_results)
transfer_average_accuracy, transfer_overall_accuracy, transfer_sum_cm = MV_Results_ByGroup.averageAccuracy(transfer_accuracy, transfer_cm)
transfer_cm_recall = MV_Results_ByGroup.confusionMatrix(transfer_sum_cm, recall=True)
print(transfer_average_accuracy, transfer_cm_recall)
print(transfer_overall_accuracy)


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