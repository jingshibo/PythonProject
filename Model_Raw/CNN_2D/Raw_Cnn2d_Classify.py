##
from Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset, Raw_Cnn2d_Model
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
import datetime

##  read sensor data and filtering

# basic information
subject = 'Shibo'
version = 1  # the data from which experiment version to process
modes = ['up_down', 'down_up']
up_down_session = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
down_up_session = [10, 11, 12, 13, 19, 24, 25, 26, 27, 28, 20]
sessions = [up_down_session, down_up_session]
feature_window_increment_ms = 32  # the window increment for feature calculation

# read and filter data
split_parameters = Preprocessing.readSplitParameters(subject, version)
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-1024, end_position=1024)
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data)
del emg_filtered_data
fold = 5  # 5-fold cross validation
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_preprocessed)
window_per_repetition = cross_validation_groups['group_0']['train_set']['emg_LWLW'][0].shape[0]  # how many windows for each event repetition
del emg_preprocessed


##  reorganize data
now = datetime.datetime.now()
sliding_window_dataset = Raw_Cnn2d_Dataset.seperateEmgData(cross_validation_groups, separation_window_size=512, increment=64)
del cross_validation_groups
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups
print(datetime.datetime.now() - now)


# ##
# import os
# import json
#
# model_type = 'Raw_Cnn2d'
# result_set = 1
#
# for group_number, group_value in shuffled_groups.items():
#     for key, value in group_value.items():
#         group_value[key] = value.tolist()
#
# data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\extracted_features'
# result_file = f'subject_{subject}_Experiment_{version}_emg_raw_data_{result_set}.json'
# result_path = os.path.join(data_dir, result_file)
#
# with open(result_path, 'w') as json_file:
#     json.dump(shuffled_groups, json_file, indent=8)



##  classify using a single cnn 2d model
now = datetime.datetime.now()
models, model_results = Raw_Cnn2d_Model.classifyUsingCnn2dModel(shuffled_groups)
print(datetime.datetime.now() - now)

## save model results
# result_set = 0
# Sliding_Ann_Results.saveModelResults(subject, model_results, version, result_set, window_per_repetition, feature_window_increment_ms, model_type='Raw_Cnn2d')


## majority vote results using prior information, with a sliding windows to get predict results at different delay points
reorganized_results = MV_Results_ByGroup.regroupModelResults(model_results)
predict_window_shift_unit = 2
sliding_majority_vote_by_group = Sliding_Ann_Results.majorityVoteResultsByGroup(reorganized_results, window_per_repetition,
    predict_window_shift_unit, initial_start=0, initial_end=16)
accuracy_bygroup, cm_bygroup = Sliding_Ann_Results.getAccuracyPerGroup(sliding_majority_vote_by_group)
# calculate the accuracy and cm. Note: the first dimension refers to each delay
average_accuracy_with_delay, overall_accuracy_with_delay, sum_cm_with_delay = MV_Results_ByGroup.averageAccuracyByGroup(accuracy_bygroup,
    cm_bygroup)
accuracy, cm_recall = Sliding_Ann_Results.getAccuracyCm(overall_accuracy_with_delay, sum_cm_with_delay, feature_window_increment_ms,
    predict_window_shift_unit)


# ##
# # tf.keras.backend.clear_session()
# regularization = tf.keras.regularizers.L2(0.00001)
# initializer = tf.keras.initializers.HeNormal()
# inputs = tf.keras.Input(shape=(512, 130, 1))
# x = tf.keras.layers.Conv2D(32, 7, dilation_rate=1, strides=2, padding='same', data_format='channels_last')(inputs)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.ReLU()(x)
# x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
# x = tf.keras.layers.Conv2D(64, 5, dilation_rate=1, strides=2, padding='same', data_format='channels_last')(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.ReLU()(x)
# x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
# x = tf.keras.layers.Conv2D(128, 3, dilation_rate=1, strides=2, padding='same', data_format='channels_last')(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.ReLU()(x)
# x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
# x = tf.keras.layers.Flatten(data_format='channels_last')(x)
# x = tf.keras.layers.Dense(1000, kernel_regularizer=regularization, kernel_initializer=initializer)(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.ReLU()(x)
# x = tf.keras.layers.Dropout(0.5)(x)
# x = tf.keras.layers.Dense(13, kernel_regularizer=regularization, kernel_initializer=initializer)(x)
# outputs = tf.keras.layers.Softmax()(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_cnn")
# # view models
# model.summary()
#
#
# ##
# from keras import backend as K
# K.clear_session()

##
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from torchinfo import summary
class ConvNet(nn.Module):
    def __init__(self, input_channel, class_number):
        super(ConvNet, self).__init__()

        # define layer parameter
        self.conv1_parameter = [32, 7]
        self.conv2_parameter = [64, 5]
        self.conv3_parameter = [128, 3]
        self.linear1_parameter = 1000
        self.linear2_parameter = 100

        # define convolutional layer
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=self.conv1_parameter[0], kernel_size=self.conv1_parameter[1], stride=1),
            nn.BatchNorm2d(self.conv1_parameter[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=self.conv1_parameter[0], out_channels=self.conv2_parameter[0], kernel_size=self.conv2_parameter[1], stride=1),
            nn.BatchNorm2d(self.conv2_parameter[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=self.conv2_parameter[0], out_channels=self.conv3_parameter[0], kernel_size=self.conv3_parameter[1], stride=1),
            nn.BatchNorm2d(self.conv3_parameter[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        # define dense layer
        self.linear_layer = nn.Sequential(
            # nn.LazyLinear(self.linear1_parameter),
            # nn.BatchNorm1d(self.linear1_parameter),
            # nn.ReLU(),
            # nn.Dropout(0.5),

            nn.LazyLinear(self.linear2_parameter),
            nn.BatchNorm1d(self.linear2_parameter),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.LazyLinear(out_features=class_number)
        )

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        if self.training is False:
            x = F.softmax(x, dim=1)
        return x


##  check model
net = ConvNet(1,13)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = net.to(device)  # move the model to GPU
print(net)

##
net = ConvNet(1,13)
summary(net, input_size=(512, 1, 512, 130))