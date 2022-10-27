## import modules
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from Models.Utility_Functions import Confusion_Matrix

##
group_results = []
for group_number, group_value in shuffled_groups.items():
    predict_results = {}
    for transition_type, transition_train_data in group_value["train_set"].items():
        # training data
        train_set_x = transition_train_data['feature_x'][:, 0:1040]
        train_set_y = transition_train_data['feature_onehot_y']
        class_number = len(set(transition_train_data['feature_int_y']))
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
        model.fit(train_set_x, train_set_y, validation_split=0.1, epochs=50, batch_size=512, verbose='auto')
        print(datetime.datetime.now() - now)

        # test data
        test_set_x = group_value['test_set'][transition_type]['feature_x'][:, 0:1040]
        test_set_y = group_value['test_set'][transition_type]['feature_onehot_y']
        # test model
        predictions = model.predict(test_set_x)  # return predicted probabilities
        predict_y = np.argmax(predictions, axis=-1)  # return predicted labels
        test_loss, test_accuracy = model.evaluate(test_set_x, test_set_y)  # return loss and accuracy values

        predict_results[transition_type] = {"true_value": group_value['test_set'][transition_type]['feature_int_y'],
            "predict_value": predict_y, "predict_accuracy": test_accuracy}
    group_results.append(predict_results)

## majority vote
bin_results = []
for each_group in group_results:
    bin_transitions = {}
    for transition_type, transition_result in each_group.items():
        true_y = []
        predict_y = []
        for key, value in transition_result.items():
            if key == 'true_value':
                for i in range(0, len(value), window_per_repetition):
                    true_y.append(value[i: i+window_per_repetition])
            elif key == 'predict_value':
                for i in range(0, len(value), window_per_repetition):
                    predict_y.append(value[i: i+window_per_repetition])
        bin_transitions[transition_type] = {"true_value": true_y, "predict_value": predict_y}
    bin_results.append(bin_transitions)

##
majority_results = []
for each_group in bin_results:
    majority_transitions = {}
    for transition_type, transition_result in each_group.items():
        true_y = []
        predict_y = []
        for key, value in transition_result.items():
            if key == 'true_value':
                true_y = [np.bincount(i).argmax() for i in value]
            elif key == 'predict_value':
                predict_y = [np.bincount(i).argmax() for i in value]
        majority_transitions[transition_type] = {"true_value": np.array(true_y), "predict_value": np.array(predict_y)}
    majority_results.append(majority_transitions)

## accuracy
test_accuracy = []
test_cm = []
for each_group in majority_results:
    transition_cm = {}
    transition_accuracy = {}
    for transition_type, transition_result in each_group.items():
        true_y = transition_result['true_value']
        predict_y = transition_result['predict_value']
        numCorrect = np.count_nonzero(true_y == predict_y)
        transition_accuracy[transition_type] = numCorrect / len(true_y) * 100
        transition_cm[transition_type] = confusion_matrix(y_true=true_y, y_pred=predict_y)
    test_accuracy.append(transition_accuracy)
    test_cm.append(transition_cm)

## avarage value
accuracy = {'transition_LW':0, 'transition_SA':0, 'transition_SD':0, 'transition_SS':0}
for value in test_accuracy:
    accuracy['transition_LW'] = accuracy['transition_LW'] + value['transition_LW']
    accuracy['transition_SA'] = accuracy['transition_SA'] + value['transition_SA']
    accuracy['transition_SD'] = accuracy['transition_SD'] + value['transition_SD']
    accuracy['transition_SS'] = accuracy['transition_SS'] + value['transition_SS']
for key, value in accuracy.items():
    accuracy[key] = value / len(test_accuracy)

cm = {'transition_LW':np.zeros((4, 4)), 'transition_SA':np.zeros((3, 3)), 'transition_SD':np.zeros((3, 3)), 'transition_SS':np.zeros((3, 3))}
for value in test_cm:
    cm['transition_LW'] = cm['transition_LW'] + value['transition_LW']
    cm['transition_SA'] = cm['transition_SA'] + value['transition_SA']
    cm['transition_SD'] = cm['transition_SD'] + value['transition_SD']
    cm['transition_SS'] = cm['transition_SS'] + value['transition_SS']
for key, value in cm.items():
    cm[key] = (np.around(value.T / np.sum(value, axis=1) * 100, decimals=1)).T

