## import modules
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from Models.Utility_Functions import Confusion_Matrix

##
results = []
for group_number, group_value in shuffled_groups.items():
    # data
    train_set_x = group_value['train_feature_x'][:, 0:1040]
    train_set_y = group_value['train_onehot_y']
    test_set_x = group_value['test_feature_x'][:, 0:1040]
    test_set_y = group_value['test_onehot_y']
    # model
    model = tf.keras.models.Sequential(name="ann_model")  # optional name
    model.add(tf.keras.layers.InputLayer(input_shape=(train_set_x.shape[1]))) # It can also be replaced by: model.add(tf.keras.Input(shape=(28,28)))
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


    # model configuration
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

    # training model
    now = datetime.datetime.now()
    model.fit(train_set_x, train_set_y, validation_split=0.1, epochs=100, batch_size=1024, verbose='auto')
    print(datetime.datetime.now() - now)

    # test model
    predictions = model.predict(test_set_x)  # return predicted probabilities
    predict_y = np.argmax(predictions, axis=-1)  # return predicted labels
    test_loss, test_accuracy = model.evaluate(test_set_x, test_set_y)  # return loss and accuracy values

    results.append({"true_value": group_value['test_int_y'], "predict_value": predict_y, "predict_accuracy": test_accuracy})


## majority vote
bin_results = []
for result in results:
    true_y = []
    predict_y = []
    for key, value in result.items():
        if key == 'true_value':
            for i in range(0, len(value), window_per_repetition):
                true_y.append(value[i: i+window_per_repetition])
        elif key == 'predict_value':
            for i in range(0, len(value), window_per_repetition):
                predict_y.append(value[i: i+window_per_repetition])
    bin_results.append({"true_value": true_y, "predict_value": predict_y})

majority_results = []
for result in results:
    true_y = []
    predict_y = []
    for key, value in result.items():
        if key == 'true_value':
            for i in range(0, len(value), window_per_repetition):
                bin_values = np.array(value[i: i+window_per_repetition])
                true_y.append(np.bincount(bin_values).argmax())
        elif key == 'predict_value':
            for i in range(0, len(value), window_per_repetition):
                bin_values = np.array(value[i: i + window_per_repetition])
                predict_y.append(np.bincount(bin_values).argmax())
    majority_results.append({"true_value": np.array(true_y), "predict_value": np.array(predict_y)})

## accuracy
cm = []
accuracy = []
for result in majority_results:
    true_y = result['true_value']
    predict_y = result['predict_value']
    numCorrect = np.count_nonzero(true_y == predict_y)
    accuracy.append(numCorrect / len(true_y) * 100)
    cm.append(confusion_matrix(y_true=true_y, y_pred=predict_y))
mean_accuracy = sum(accuracy) / len(accuracy)
sum_cm = np.sum(np.array(cm), axis=0)
print(mean_accuracy)


## plot confusion matrix
cm_recall = (np.around(sum_cm.T / np.sum(sum_cm, axis=1) * 100, decimals=1)).T
# the labels in the classes list should correspond to the one hot labels
# class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'LW', 'SALW', 'SASA', 'SASS', 'SA', 'SDLW', 'SDSD', 'SDSS', 'SD', 'SSLW', 'SSSA', 'SSSD']
class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'SALW', 'SASA', 'SASS', 'SDLW', 'SDSD', 'SDSS', 'SSLW', 'SSSA', 'SSSD']
Confusion_Matrix.plotConfusionMatrix(cm_recall, class_labels)

# sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=class_labels, xticklabels=class_labels)
# ##
# plt.imshow(cm)

##
x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(1, 4.0)
x1 / x2
