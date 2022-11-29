'''
load models pretrained using all dataset. Then train the pretrained model again using four separate groups of dataset individually.
'''


## import modules
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Models.Basic_Ann.Functions import Grouped_Ann_Dataset, Ann_Model
import datetime
import tensorflow as tf


## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use
fold = 5  # 5-fold cross validation

# read feature data
emg_features, emg_feature_reshaped = Data_Preparation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Preparation.removeSomeSamples(emg_features)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[0]  # how many windows there are for each event repetition
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_feature_data)

# reorganize data
transition_grouped = Data_Preparation.separateGroups(cross_validation_groups)
combined_groups = Grouped_Ann_Dataset.combineIntoDataset(transition_grouped, window_per_repetition)
normalized_groups = Grouped_Ann_Dataset.normalizeDataset(combined_groups)
shuffled_groups = Grouped_Ann_Dataset.shuffleTrainingSet(normalized_groups)

## load pretrained model
fold = 5
type = 1  # pre-trained model type
models = []
for number in range(fold):
    model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models_{type}\cross_validation_set_{number}'
    models.append(tf.keras.models.load_model(model_dir))

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

## train transferred models
now = datetime.datetime.now()
model_results = Ann_Model.classifyTransferAnnModel(shuffled_groups, models)
print(datetime.datetime.now() - now)

## majority vote results
majority_results = MV_Results_ByGroup.majorityVoteResults(model_results, window_per_repetition)
accuracy, cm = MV_Results_ByGroup.getAccuracyPerGroup(majority_results)
average_accuracy, sum_cm = MV_Results_ByGroup.averageAccuracy(accuracy, cm)
cm_recall = MV_Results_ByGroup.confusionMatrix(sum_cm, recall=True)
print(average_accuracy, cm_recall)

## mean accuracy for all groups
overall_accuracy = (average_accuracy['transition_LW'] * 1.5 + average_accuracy['transition_SA'] +
                    average_accuracy['transition_SD'] + average_accuracy['transition_SS']) / 4.5
print(overall_accuracy)

# ## save trained models
# type = 1  # define the type of trained model to save
# for number, model in enumerate(model_results):
#     model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models_{type}\cross_validation_set_{number}'
#     if os.path.isfile(model_dir) is False:  # check if the location is a file or directory. Only save when it is a directory
#         model['model'].save(model_dir)  # save the model to the directory