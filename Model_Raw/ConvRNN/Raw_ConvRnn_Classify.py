##
from Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset
from Models.Utility_Functions import Data_Preparation
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
from Model_Raw.ConvRNN.Functions import Raw_ConvRnn_Dataset, Raw_ConvRnn_Model
import datetime


##  read sensor data and filtering
# basic information
subject = 'Number5'
version = 0  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
down_up_session = [0, 1, 2, 3, 4, 5, 6, 8, 9]
# up_down_session = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# down_up_session = [10, 11, 12, 13, 19, 24, 25, 26, 27, 28, 20]
sessions = [up_down_session, down_up_session]


## read and filter data
split_parameters = Preprocessing.readSplitParameters(subject, version)
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-900,
    end_position=800, notchEMG=False, reordering=True)
down_sampling = True
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data, is_down_sampling=down_sampling)
del emg_filtered_data
fold = 5  # 5-fold cross validation
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_preprocessed)
del emg_preprocessed


##  define windows
predict_window_ms = 450
feature_window_ms = 350
sample_rate = 1 if down_sampling is True else 2
predict_window_size = predict_window_ms * sample_rate
feature_window_size = feature_window_ms * sample_rate
predict_window_increment_ms = 20
feature_window_increment_ms = 20
predict_window_shift_unit = int(predict_window_increment_ms / feature_window_increment_ms)
predict_of_window_number = int((predict_window_size - feature_window_size) / (feature_window_increment_ms * sample_rate))
endtime_after_toeoff_ms = 400
predict_window_per_repetition = int(endtime_after_toeoff_ms / predict_window_increment_ms) + 1


##  reorganize data
now = datetime.datetime.now()
sliding_window_dataset, feature_window_per_repetition = Raw_Cnn2d_Dataset.separateEmgData(cross_validation_groups, feature_window_size,
    increment=feature_window_increment_ms * sample_rate)
del cross_validation_groups
normalized_groups = Raw_ConvRnn_Dataset.combineNormalizedDataset(sliding_window_dataset)
del sliding_window_dataset
shuffled_groups = Raw_ConvRnn_Dataset.shuffleTrainingSet(normalized_groups, predict_window_per_repetition, predict_window_shift_unit,
    predict_of_window_number)
del normalized_groups
print(datetime.datetime.now() - now)


##  classify using a single cnn-rnn model
num_epochs = 20
batch_size = 256
decay_epochs = 10
now = datetime.datetime.now()
# train_model = Raw_Cnn2d_Model.ModelTraining(num_epochs, batch_size)
train_model = Raw_ConvRnn_Model.ModelTraining(num_epochs, batch_size, report_period=10)
models, model_results = train_model.trainModel(shuffled_groups, decay_epochs)
print(datetime.datetime.now() - now)


##  evaluate the prediction results
