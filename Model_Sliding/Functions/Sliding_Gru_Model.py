'''
classifying using GRU models at different timestamps
'''


## import modules
import datetime
import numpy as np
import tensorflow as tf
from Model_Sliding.Functions import Sliding_Gru_Dataset
import copy
import json
import os


## a "many to one" GRU model with sliding windows that returns only the last output
def classifySlidingGtuLastOneModel(shuffled_groups):
    group_results = []
    group_models = []
    for group_number, group_value in shuffled_groups.items():

        # select channels to calculate
        channel_to_compute = 'emg_all'
        train_set, test_set = Sliding_Gru_Dataset.select1dFeatureChannels(group_value, channel_to_compute)
        class_number = len(set(train_set['train_int_y']))

        # layer parameters
        regularization = tf.keras.regularizers.L2(0.00001)
        initializer = tf.keras.initializers.HeNormal()

        # model structure
        model = tf.keras.models.Sequential(name="gru_model")  # optional name
        model.add(tf.keras.layers.GRU(1000, return_sequences=True, dropout=0.5, input_shape=(None, train_set['train_set_x'].shape[2])))
        model.add(tf.keras.layers.GRU(1000, return_sequences=False, dropout=0.5))
        model.add(tf.keras.layers.Dense(600, kernel_regularizer=regularization, kernel_initializer=initializer))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(class_number))
        model.add(tf.keras.layers.Softmax())  # or activation=tf.nn.softmax
        # view model
        model.summary()

        # model parameters
        num_epochs = 50
        decay_epochs = 30
        batch_size = 1024
        decay_steps = decay_epochs * len(train_set['train_set_y']) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=decay_steps, decay_rate=0.5)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-08)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
        # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

        # train model
        now = datetime.datetime.now()
        model.fit(train_set['train_set_x'], train_set['train_set_y'], validation_split=0.1, epochs=num_epochs, batch_size=batch_size, shuffle=True, verbose='auto')
        print(datetime.datetime.now() - now)

        # test model
        test_results_per_shift = {}
        trained_model_per_shift = {}
        for shift_number, shift_value in test_set.items():
            predictions = model.predict(shift_value['test_set_x'])  # return predicted probabilities
            predict_y = np.argmax(predictions, axis=-1)  # return predicted labels
            test_loss, test_accuracy = model.evaluate(shift_value['test_set_x'], shift_value['test_set_y'])  # return loss and accuracy values

            test_results_per_shift[shift_number] = {"true_value": shift_value['test_int_y'], "predict_softmax": predictions,
                "predict_value": predict_y, "predict_accuracy": test_accuracy}
            trained_model_per_shift[shift_number] = model

        group_results.append(test_results_per_shift)
        group_models.append(trained_model_per_shift)

    return group_models, group_results


##  save the classification results to disk
def saveModelResults(subject, model_results, version, result_set):
    results = copy.deepcopy(model_results)
    for result in results:
        for shift_number, shift_value in result.items():
            shift_value['true_value'] = shift_value['true_value'].values.tolist()
            shift_value['predict_softmax'] = shift_value['predict_softmax'].values.tolist()
            shift_value['predict_value'] = shift_value['predict_value'].tolist()

    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\model_results'
    result_file = f'subject_{subject}_Experiment_{version}_model_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    with open(result_path, 'w') as json_file:
        json.dump(results, json_file, indent=8)


##  read the classification results from disk
def loadModelResults(subject, version, result_set):
    data_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\model_results'
    result_file = f'subject_{subject}_Experiment_{version}_model_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    # read json file
    with open(result_path) as json_file:
        result_json = json.load(json_file)
    for result in result_json:
        for shift_number, shift_value in result.items():
            shift_value['true_value'] = np.array(shift_value['true_value'])
            shift_value['predict_softmax'] = np.array(shift_value['predict_softmax'])
            shift_value['predict_value'] = np.array(shift_value['predict_value'])

    return result_json