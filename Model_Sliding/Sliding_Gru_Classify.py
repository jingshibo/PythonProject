## import modules
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.Functions import Sliding_Gru_Dataset, Sliding_Gru_Model, Results_ByGroup
import datetime

## read emg data
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 0  # which feature set to use
fold = 5  # 5-fold cross validation

# read feature data
emg_features, emg_feature_2d = Data_Preparation.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Data_Preparation.removeSomeSamples(emg_features)  # only remove some modes here
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_feature_data)

# create dataset
emg_sliding_features = Sliding_Gru_Dataset.createSlidingDataset(cross_validation_groups, initial_start=0, initial_end=16, shift=4)
window_per_repetition = emg_sliding_features['group_0']['train_set']['emg_LWLW_features'][0].shape[0]  # how many windows for each event repetition
normalized_groups = Sliding_Gru_Dataset.combineNormalizedDataset(emg_sliding_features, window_per_repetition)
shuffled_groups = Sliding_Gru_Dataset.shuffleTrainingSet(normalized_groups)


## classify using a "many to one" GRU model
now = datetime.datetime.now()
model_results = Sliding_Gru_Model.classifySlidingGtuLastOneModel(shuffled_groups)
print(datetime.datetime.now() - now)


##
import matplotlib.pyplot as plt
accuracy = []
for result in model_results:
    for shift_number, shift_value in result.items():
        accuracy.append(shift_value['predict_accuracy'])
plt.plot(accuracy)


## results using prior information (no majority vote used, what we need here is the grouped accuracy calculation)
regrouped_results = Results_ByGroup.regroupModelResults(model_results)

##
reorganized_softmax, reorganized_prediction, reorganized_true = Results_ByGroup.reorganizePredictValues(regrouped_results)



##
accuracy_with_prior, cm_with_prior = Results_ByGroup.getAccuracyPerGroup(regrouped_results)
average_accuracy, overall_accuracy, sum_cm = Results_ByGroup.averageAccuracy(accuracy_with_prior, cm_with_prior)
cm_recall = MV_Results_ByGroup.confusionMatrix(sum_cm, recall=True)
print(overall_accuracy)




## the problem is that it takes up a lot memories
# import numpy as np
# import tensorflow as tf
# length3 = np.random.uniform(0, 1, size=(3, 2))
# length4 = np.random.uniform(0, 1, size=(4, 2))
# tf.keras.preprocessing.sequence.pad_sequences([length3, length4], dtype='float32', padding='post')
