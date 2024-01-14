##
from Bipolar_EMG.Utility_Functions import Emg_Imu_Preprocessing
from Bipolar_EMG.Models import Dataset_Model
from Transition_Prediction.Models.Utility_Functions import Data_Preparation
from Transition_Prediction.Models.ANN.Functions import Ann_Dataset
import datetime


## selecting certain inputs
sensor_sets = {
    # 'emg_0': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM'],  # three front and back
    'emg_1': ['RF', 'TA', 'BF', 'GM'],  # front two + back two
    # 'emg_2': ['RF', 'TA', 'VM'],  # all front
    # 'emg_3': ['BF', 'SL', 'GM'],  # all back
    'emg_4': ['RF', 'BF', 'VM'],  # upper three
    'emg_5': ['TA', 'SL', 'GM'],  # lower three
    'emg_6': ['RF', 'TA'],  # front two
    'emg_7': ['BF', 'GM'],  # back two
    'emg_8': ['RF', 'BF'],  # upper two
    'emg_9': ['TA', 'GM'],  # lower two
    # 'emg_10': ['RF'],
    # 'emg_11': ['TA'],
    # 'emg_12': ['BF'],
    # 'emg_13': ['SL'],
    # 'emg_14': ['VM'],
    # 'emg_15': ['GM'],
    # # only imu
    # 'imu_0': ['LL', 'FT', 'UL'],
    # 'imu_1': ['FT', 'UL'],
    # 'imu_2': ['LL', 'UL'],
    # 'imu_3': ['LL'],
    # 'imu_4': ['FT'],
    # 'imu_5': ['UL'],
    # # emg+imu
    # 'emg_imu_0': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM', 'LL', 'FT', 'UL'],
    # 'emg_imu_1': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM', 'LL', 'UL'],
    'emg_imu_2': ['RF', 'TA', 'BF', 'SL', 'VM', 'GM', 'FT'],
    'emg_imu_3': ['RF', 'TA', 'BF', 'GM', 'LL', 'FT', 'UL'],
    'emg_imu_4': ['RF', 'TA', 'BF', 'GM', 'LL', 'UL'],
    'emg_imu_5': ['RF', 'TA', 'BF', 'GM', 'FT'],
}
emg = {'RF': list(range(0, 48, 6)), 'VM': list(range(1, 48, 6)), 'TA': list(range(2, 48, 6)), 'BF': list(range(3, 48, 6)),
    'GM': list(range(4, 48, 6)), 'SL': list(range(5, 48, 6))}
imu = {sensor: [i for j in range(4) for i in range(start + 18 * j, start + 6 + 18 * j)] for sensor, start in
    zip(['LL', 'FT', 'UL'], range(48, 61, 6))}  # '4' in the expression is the number of features extracted for each measurement


## read and cross validate dataset
# read data
subjects = ['Number1', 'Number2', 'Number3', 'Number5', 'Number7', 'Number9', 'Number10']
# subjects = ['Number1']
feature_set = 0

for subject in subjects:
    emg_features, imu_features = Emg_Imu_Preprocessing.readFeatures(subject, feature_set)
    emg_imu_combined = {key: [emg_features + imu_features for imu_features, emg_features in zip(imu_features[key], emg_features[key])]
        for key in imu_features}  # combine emg and imu data together

    # cross validation sets
    emg_imu_cross_validation = Data_Preparation.crossValidationSet(5, emg_imu_combined, shuffle=False)

    ## training model
    now = datetime.datetime.now()
    for sensor_set in sensor_sets.values():
        # select sensors as input
        selected_dataset = sum([emg.get(sensor, []) + imu.get(sensor, []) for sensor in sensor_set], [])
        input_cross_validation = Dataset_Model.selectInput(emg_imu_cross_validation, selected_dataset)

        # shuffle normalized dataset
        input_normalized = Dataset_Model.combineNormalizedDataset(input_cross_validation)
        input_shuffled_groups = Ann_Dataset.shuffleTrainingSet(input_normalized)

        ## classify using a single ann model
        models, model_results = Dataset_Model.classifyUsingAnnModel(input_shuffled_groups)

        ## predict MV results
        predict_results, true_labels = Dataset_Model.reorganizePredictionResults(model_results)
        predict_mv_results = [{mode: Dataset_Model.majority_vote(value, n=5) for mode, value in group_result.items()} for group_result in predict_results]
        average_accuracy, average_cm_number, average_cm_recall = Dataset_Model.calculateAccuracy(predict_mv_results, true_labels)

        ## save results
        model_type = '+'.join(sensor_set)
        result_set = 0
        Dataset_Model.saveResult(subject, average_accuracy, average_cm_number, average_cm_recall, model_type, result_set, project='Bipolar_Data')
    print(datetime.datetime.now() - now)


