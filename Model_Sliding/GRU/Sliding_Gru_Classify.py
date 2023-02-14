'''
classify using a GRU model with sliding predict windows. if the first selected window does not generate a softmax probability greater
than the treashold, move to the next window until the final window.
'''


## import modules
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.GRU.Functions import Sliding_Evaluation_ByGroup, Sliding_Gru_Dataset, Sliding_Gru_Model, Sliding_Results_ByGroup
import datetime


## read emg data
# basic information
subject = "Number3"
version = 0  # which experiment data to process
feature_set = 0  # which feature set to use
fold = 5  # 5-fold cross validation

# window parameters
predict_window_ms = 400
feature_window_ms = 300
sample_rate = 2
predict_window_size = predict_window_ms * sample_rate
feature_window_size = feature_window_ms * sample_rate
predict_window_increment_ms = 20
feature_window_increment_ms = 20
predict_window_shift_unit = int(predict_window_increment_ms / feature_window_increment_ms)
predict_of_window_number = int((predict_window_size - feature_window_size) / (feature_window_increment_ms * sample_rate)) + 1

## read feature data
emg_features, emg_feature_2d = Data_Preparation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Preparation.removeSomeSamples(emg_features)  # only remove some modes here
del emg_features, emg_feature_2d,
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_feature_data)
del emg_feature_data  # to release memory

##
# create dataset
now = datetime.datetime.now()
emg_sliding_features = Sliding_Gru_Dataset.createSlidingDataset(cross_validation_groups, predict_window_shift_unit, initial_start=0,
    predict_of_window_number=predict_of_window_number)
# del cross_validation_groups
##
feature_window_per_repetition = emg_sliding_features['group_0']['train_set']['emg_LWLW_features'][0].shape[0]  # how many windows for each event repetition
normalized_groups = Sliding_Gru_Dataset.combineNormalizedDataset(emg_sliding_features, feature_window_per_repetition)
# del emg_sliding_features
shuffled_groups = Sliding_Gru_Dataset.shuffleTrainingSet(normalized_groups)
# del normalized_groups  # to release memory
print(datetime.datetime.now() - now)


## classify using a "many to one" GRU model
now = datetime.datetime.now()
models, model_results = Sliding_Gru_Model.classifySlidingGtuLastOneModel(shuffled_groups)
print(datetime.datetime.now() - now)
del shuffled_groups  # remove the variable
# save model results
result_set = 0
Sliding_Gru_Model.saveModelResults(subject, model_results, version, result_set, feature_window_increment_ms, predict_window_shift_unit)


##  group the classification results together, starting from diffferent initial_predict_time settings, ending at the given end_predict_time
end_predict_time = int(500/32) + 1  # define the end prediction timestamp at which the predict end
group_sliding_results = Sliding_Evaluation_ByGroup.groupSlidingResults(model_results, predict_window_shift_unit,
    feature_window_increment_ms, end_predict_time, threshold=0.999)
accuracy = {}
for key, value in group_sliding_results.items():
    accuracy[key] = value['overall_accuracy']


##  print accuracy results
# eng_point = (len(accuracy)-1) * 32
# x_label = [f'{i*32}~{eng_point}ms' for i in range(len(accuracy))]
# y_value = [round(i, 1) for i in list(accuracy.values())]
#
# fig, ax = plt.subplots()
# bars = ax.bar(range(len(accuracy)), y_value)
# ax.set_xticks(range(len(accuracy)), x_label)
# ax.bar_label(bars)
# ax.set_ylim([93, 100])
# ax.set_xlabel('prediction time range')
# ax.set_ylabel('prediction accuracy(%)')
# plt.title('Prediction accuracy for the prediction starting from different time points')


## (separately) predict results at a certain initial_predict_time (category + delay)
# regroup the model results using prior information (no majority vote used, what we need here is the grouped accuracy calculation)
regrouped_results = Sliding_Results_ByGroup.regroupModelResults(model_results)
# reorganize the regrouped model results based on the timestamps
reorganized_softmax, reorganized_prediction, reorganized_truevalues = Sliding_Results_ByGroup.reorganizePredictValues(regrouped_results)
# only keep the results after the initial_predict_time
initial_predict_time = int(0/32)  # define the initial prediction timestamp from which the predict starts
end_predict_time = int(500/32) + 1  # define the end prediction timestamp at which the predict end
reduced_softmax, reduced_prediction = Sliding_Results_ByGroup.reducePredictResults(reorganized_softmax, reorganized_prediction, initial_predict_time, end_predict_time)
#  find the first timestamps at which the softmax value is larger than the threshold
first_timestamps = Sliding_Results_ByGroup.findFirstTimestamp(reduced_softmax, threshold=0.999)
# get the predict results based on timestamps from the reorganized_prediction table and convert the timestamp to delay(ms)
sliding_prediction = Sliding_Results_ByGroup.getSlidingPredictResults(reduced_prediction, first_timestamps, initial_predict_time, predict_window_shift_unit, feature_window_increment_ms)


## evaluate the prediction results at a certain initial_predict_time
# calculate the prediction accuracy
accuracy_bygroup, cm_bygroup = Sliding_Evaluation_ByGroup.getAccuracyPerGroup(sliding_prediction, reorganized_truevalues)
average_accuracy, overall_accuracy, sum_cm = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup, cm_bygroup)
cm_recall = MV_Results_ByGroup.confusionMatrix(sum_cm, is_recall=True)
print(overall_accuracy)
# calculate the prediction delay (separately for those with correct or false prediction results)
correct_results, false_results = Sliding_Evaluation_ByGroup.integrateResults(sliding_prediction, reorganized_truevalues)
predict_delay_category, predict_delay_overall, mean_delay, std_delay = Sliding_Evaluation_ByGroup.countDelay(correct_results)
false_delay_category, false_delay_overall, false_delay_mean, false_delay_std = Sliding_Evaluation_ByGroup.countDelay(false_results)
predict_delay_overall, predict_delay_category, false_delay_overall, false_delay_category = Sliding_Evaluation_ByGroup.delayAccuracy(
    predict_delay_overall, predict_delay_category, false_delay_overall, false_delay_category)


# ## save model results
# type = 1  # define the type of trained model to save
# for number, model in enumerate(model_results):
#     model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models_{type}\cross_validation_set_{number}'
#     if os.path.isfile(model_dir) is False:  # check if the location is a file or directory. Only save when it is a directory
#         model['model'].save(model_dir)  # save the model to the directory
#
#
# ## load trained model
# fold = 5
# type = 1  # pre-trained model type
# models = []
# for number in range(fold):
#     model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models_{type}\cross_validation_set_{number}'
#     models.append(tf.keras.models.load_model(model_dir))
#
#
# ## save trained models
# import os
# type = 1  # define the type of trained model to save
# for number, model in enumerate(model_results):
#     model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\model_result_{type}\cross_validation_set_{number}'
#     if os.path.isfile(model_dir) is False:  # check if the location is a file or directory. Only save when it is a directory
#         model.save(model_dir)  # save the model to the directory
#
#
# ## save and read shuffled_groups
# import os
# data_set = 0
# data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\model_dataset'
# feature_file = f'subject_{subject}_Experiment_{version}_model_dataset_{data_set}.npy'
# feature_path = os.path.join(data_dir, feature_file)
# ##
# np.save(feature_path, shuffled_groups)
# ##
# shift_unit = 2
# shuffled_groups = np.load(feature_path, allow_pickle=True).item()





