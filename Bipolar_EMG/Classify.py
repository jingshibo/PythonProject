##
from Bipolar_EMG.Utility_Functions import Emg_Imu_Preprocessing
from Bipolar_EMG.Models import Dataset_Model
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Transition_Prediction.Models.ANN.Functions import Ann_Dataset
import datetime


## read and cross validate dataset
# read data
subject = 'Number1'
feature_set = 0
emg_features, imu_features = Emg_Imu_Preprocessing.readFeatures(subject, feature_set)
emg_imu_combined = {key: [emg_features + imu_features for imu_features, emg_features in zip(imu_features[key], emg_features[key])]
    for key in imu_features}  # combine emg and imu data together

# cross validation sets
emg_imu_cross_validation = Data_Preparation.crossValidationSet(5, emg_imu_combined, shuffle=False)
emg = {'RF': list(range(0, 8)), 'VM': list(range(8, 16)), 'TA': list(range(16, 24)), 'BF': list(range(24, 32)), 'GM': list(range(32, 40)), 'SL': list(range(40, 48))}
imu = {'LL': list(range(48, 72)), 'FT': list(range(72, 96)), 'UL': list(range(96, 120))}


## selecting certain inputs
sensor_set_0 = ['RF', 'TA', 'BF', 'SL', 'VM', 'GM']
sensor_set_1 = ['RF', 'TA', 'BF', 'SL']
sensor_set_2 = ['RF', 'TA']
sensor_set_3 = ['BF', 'SL']
sensor_set_4 = ['RF', 'TA', 'VM']
sensor_set_5 = ['BF', 'SL', 'GM']
sensor_set_6 = ['RF', 'TA']
sensor_set_7 = ['RF']

# organize the dataset
selected_dataset = sum([emg.get(sensor, []) + imu.get(sensor, []) for sensor in sensor_set_7], [])
input_cross_validation = Dataset_Model.selectInput(emg_imu_cross_validation, selected_dataset)


## shuffle combined cross validation data set
input_normalized = Dataset_Model.combineNormalizedDataset(input_cross_validation)
input_shuffled_groups = Ann_Dataset.shuffleTrainingSet(input_normalized)


## classify using a single ann model
now = datetime.datetime.now()
models, model_results = Dataset_Model.classifyUsingAnnModel(input_shuffled_groups)
print(datetime.datetime.now() - now)


## predict MV results
predict_results, true_labels = Dataset_Model.reorganizePredictionResults(model_results)
predict_mv_results = [{mode: Dataset_Model.majority_vote(value, n=5) for mode, value in group_result.items()} for group_result in predict_results]
average_accuracy, average_cm_number, average_cm_recall = Dataset_Model.calculateAccuracy(predict_mv_results, true_labels)


## save results
model_type = '+'.join(sensor_set_4)
result_set = '1'
Dataset_Model.saveResult(subject, average_accuracy, average_cm_number, average_cm_recall, model_type, result_set, project='Bipolar_Data')


