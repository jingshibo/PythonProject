##
import datetime
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from Processing.Utility_Functions import Feature_Storage, Data_Reshaping
from Models.Utility_Functions import Confusion_Matrix

## input emg data
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 0  # which feature set to use
emg_feature_data = Feature_Storage.readEmgFeatures(subject, version, feature_set)
# if you need to use CNN model, you need to reshape the data
emg_feature_reshaped = Data_Reshaping.reshapeEmgFeatures(emg_feature_data)
# abandon samples from some modes
emg_feature_sample_reduced = copy.deepcopy(emg_feature_data)
emg_feature_sample_reduced['emg_LWLW_features'] = emg_feature_sample_reduced['emg_LWLW_features'][
int(emg_feature_data['emg_LWLW_features'].shape[0] / 4): int(emg_feature_data['emg_LWLW_features'].shape[0] * 3 / 4), :]
emg_feature_sample_reduced.pop('emg_LW_features', None)
emg_feature_sample_reduced.pop('emg_SD_features', None)
emg_feature_sample_reduced.pop('emg_SA_features', None)

# emg_feature_sample_reduced['emg_LW_features'] = emg_feature_sample_reduced['emg_LW_features'][
# int(emg_feature_data['emg_LW_features'].shape[0] / 4): int(emg_feature_data['emg_LW_features'].shape[0] * 3 / 4), :]
# emg_feature_sample_reduced['emg_SD_features'] = emg_feature_sample_reduced['emg_SD_features'][
# int(emg_feature_data['emg_SD_features'].shape[0] / 2):, :]
# emg_feature_sample_reduced['emg_SA_features'] = emg_feature_sample_reduced['emg_SA_features'][
# int(emg_feature_data['emg_SA_features'].shape[0] / 2):, :]


##  class name to labels  # according to the alphabetical order
class_all = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_LW_features': 4,
    'emg_SALW_features': 5, 'emg_SASA_features': 6, 'emg_SASS_features': 7, 'emg_SA_features': 8, 'emg_SDLW_features': 9,
    'emg_SDSD_features': 10, 'emg_SDSS_features': 11, 'emg_SD_features': 12, 'emg_SSLW_features': 13, 'emg_SSSA_features': 14,
    'emg_SSSD_features': 15}

class_reduced = {'emg_LWLW_features': 0, 'emg_LWSA_features': 1, 'emg_LWSD_features': 2, 'emg_LWSS_features': 3, 'emg_SALW_features': 4,
    'emg_SASA_features': 5, 'emg_SASS_features': 6, 'emg_SDLW_features': 7, 'emg_SDSD_features': 8, 'emg_SDSS_features': 9,
    'emg_SSLW_features': 10, 'emg_SSSA_features': 11, 'emg_SSSD_features': 12}


## put all data into a dataset
emg_feature_x = []
emg_feature_y = []
class_number = len(emg_feature_sample_reduced.keys())
for gait_event_label, gait_event_emg in emg_feature_sample_reduced.items():
    emg_feature_x.extend(gait_event_emg)
    emg_feature_y.extend([gait_event_label] * len(gait_event_emg))
emg_feature_x = np.array(emg_feature_x)
emg_feature_y = np.array(emg_feature_y)


## only use emg data before gait events
# emg_x = emg_feature_x[0::17, :]
# emg_y = emg_feature_y[0::17]
# emg_feature_x = emg_x

##  only use selected emg channels
# emg_feature_tibialis_x = emg_feature_x[:, 0: 65]
# for i in range(1, 8):
#     emg_feature_tibialis_x = np.concatenate((emg_feature_tibialis_x, emg_feature_x[:, 0+130*i: 65+130*i]), axis=1)
# emg_feature_rectus_x = emg_feature_x[:, 65: 130]
# for i in range(1, 8):
#     emg_feature_rectus_x = np.concatenate((emg_feature_rectus_x, emg_feature_x[:, 65+130*i: 130+130*i]), axis=1)
# emg_feature_bipolar_x = emg_feature_x[:, 33].reshape(len(emg_feature_y), 1)
# for i in range(1, 16):
#     emg_feature_bipolar_x = np.concatenate((emg_feature_bipolar_x, emg_feature_x[:, 33+65*i].reshape(len(emg_feature_y), 1)), axis=1)
# emg_feature_x = emg_feature_tibialis_x


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
x_train, y_train_onehot, y_train_int = emg_feature_x[train_idx, :], onehot_categories_y[train_idx, :], int_categories_y[train_idx]
x_test, y_test_onehot, y_test_int = emg_feature_x[test_idx, :], onehot_categories_y[test_idx, :], int_categories_y[test_idx]


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
model.add(tf.keras.layers.Dense(class_number))
model.add(tf.keras.layers.Softmax())

# view model
model.summary()
model.layers


## model configuration
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

## training model
now = datetime.datetime.now()
model.fit(x_train_norm, y_train_onehot, validation_split=0.2, epochs=200, batch_size=1024, verbose='auto')
print(datetime.datetime.now() - now)

## test model
predictions = model.predict([x_test_norm])  # return predicted probabilities
y_pred = np.argmax(predictions, axis=-1)  # return predicted labels
test_loss, test_acc = model.evaluate(x_test_norm, y_test_onehot)  # return loss and accuracy values


## plot confusion matrix
cm = confusion_matrix(y_true=y_test_int, y_pred=y_pred)
# the labels in the classes list should correspond to the one hot labels
# class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'LW', 'SALW', 'SASA', 'SASS', 'SA', 'SDLW', 'SDSD', 'SDSS', 'SD', 'SSLW', 'SSSA', 'SSSD']
class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'SALW', 'SASA', 'SASS', 'SDLW', 'SDSD', 'SDSS', 'SSLW', 'SSSA', 'SSSD']
Confusion_Matrix.plotConfusionMatrix(cm, class_labels)

##
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=class_labels, xticklabels=class_labels)
##
plt.imshow(cm)