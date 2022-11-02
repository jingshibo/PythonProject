## import modules
from Models.Basic_Ann.Functions import Ann_Dataset
from Models.CNN.Functions import Cnn_Dataset

## read and cross validate dataset
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use
fold = 5  # 5-fold cross validation

# read feature data
emg_features, emg_feature_reshaped = Ann_Dataset.loadEmgFeature(subject, version, feature_set)
emg_feature_data = Ann_Dataset.removeSomeMode(emg_feature_reshaped)
window_per_repetition = emg_feature_data['emg_LWLW_features'][0].shape[-1]  # how many windows there are for each event repetition

# reorganize data
cross_validation_groups = Ann_Dataset.crossValidationSet(fold, emg_feature_data)
normalized_groups = Cnn_Dataset.combineNormalizedDataset(cross_validation_groups, window_per_repetition)
shuffled_groups = Cnn_Dataset.shuffleTrainingSet(normalized_groups)


## classify using cnn models