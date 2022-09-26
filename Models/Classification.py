##
import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Processing.Utility_Functions import Feature_Storage, Data_Reshaping
from sklearn.preprocessing import LabelEncoder



## input emg data
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 0  # which features to select
emg_feature_data = Feature_Storage.readEmgFeatures(subject, version, feature_set)
# if you need to use CNN model, you need to reshape the data
emg_feature_reshaped = Data_Reshaping.reshapeEmgFeatures(emg_feature_data)


## put all data into a dataset
emg_feature_x = []
emg_feature_y = []
classes = len(emg_feature_data.keys())
for gait_event_label, gait_event_emg in emg_feature_data.items():
    emg_feature_x.extend(gait_event_emg)
    emg_feature_y.extend([gait_event_label] * len(gait_event_emg))
emg_feature_x = np.array(emg_feature_x)
emg_feature_y = np.array(emg_feature_y)


## encode categories
int_categories_y = LabelEncoder().fit_transform(emg_feature_y)  # according to the alphabetical order
onehot_categories_y = tf.keras.utils.to_categorical(int_categories_y)


## shuffle data
ratio = 0.8
data_number = len(emg_feature_x)
# Shuffles the indices
idx = np.arange(data_number)
np.random.shuffle(idx)
# Uses first 80 random indices for train
train_idx = idx[: int(data_number * ratio)]
# Uses the remaining indices for validation
test_idx = idx[int(data_number * ratio):]
# Generates train and validation sets
x_train, y_train = emg_feature_x[train_idx, :], onehot_categories_y[train_idx, :]
x_test, y_test = emg_feature_x[test_idx, :], onehot_categories_y[test_idx, :]


## normalization
x_train_norm = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
x_test_norm = (x_test - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)


## model
model = tf.keras.models.Sequential(name="my_model")  # optional name
model.add(tf.keras.layers.InputLayer(input_shape=(np.shape(x_train_norm)[1]))) # It can also be replaced by: model.add(tf.keras.Input(shape=(28,28)))
model.add(tf.keras.layers.Dense(200))  # or activation=tf.nn.relu
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(200))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(200))  # or activation=tf.nn.softmax
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(classes))
model.add(tf.keras.layers.Softmax())


## view model
model.summary()
model.layers


## model configuration
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')


## training model
now = datetime.datetime.now()
model.fit(x_train_norm, y_train, validation_split=0.1, epochs=1000, batch_size=1024, verbose= 0)
print(datetime.datetime.now() - now)

## test model
val_loss, val_acc = model.evaluate(x_test_norm, y_test) # return loss and accuracy values
print(val_loss, val_acc)

