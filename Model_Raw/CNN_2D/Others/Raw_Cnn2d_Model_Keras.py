## import modules
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from Models.Utility_Functions import Confusion_Matrix
import torch

## a single CNN model
def classifyUsingCnn2dModel(shuffled_groups):
    '''
    A basic CNN model
    '''

    models = []
    results = []

    for group_number, group_value in shuffled_groups.items():
        # one muscle / bipolar data
        # train_tibialis_x = group_value['train_feature_x'][:, 0: 65]
        # test_tibialis_x = group_value['test_feature_x'][:, 0: 65]
        # train_rectus_x = group_value['train_feature_x'][:, 65: 130]
        # test_rectus_x = group_value['test_feature_x'][:, 65: 130]
        # for i in range(1, 8):
        #     train_tibialis_x = np.concatenate((train_tibialis_x, group_value['train_feature_x'][:, 0+130*i: 65+130*i]), axis=1)
        #     test_tibialis_x = np.concatenate((test_tibialis_x, group_value['test_feature_x'][:, 0 + 130 * i: 65 + 130 * i]), axis=1)
        #     train_rectus_x = np.concatenate((train_rectus_x, group_value['train_feature_x'][:, 65 + 130 * i: 130 + 130 * i]), axis=1)
        #     test_rectus_x = np.concatenate((test_rectus_x, group_value['test_feature_x'][:, 65 + 130 * i: 130 + 130 * i]), axis=1)
        # train_bipolar_x = group_value['train_feature_x'][:, 33].reshape(len(emg_feature_y), 1)
        # for i in range(1, 16):
        #     emg_feature_bipolar_x = np.concatenate((emg_feature_bipolar_x, group_value[:, 33+65*i].reshape(len(emg_feature_y), 1)),
        #     axis=1)
        # train_set_x = train_rectus_x[:, 0:520]
        # train_set_y = group_value['train_onehot_y']
        # test_set_x = test_rectus_x[:, 0:520]
        # test_set_y = group_value['test_onehot_y']

        # input data
        train_set_x = np.transpose(group_value['train_feature_x'][:, :, :, :], (3, 0, 1, 2))  # data_format:'channels_last'
        train_set_y = group_value['train_onehot_y']
        test_set_x = np.transpose(group_value['test_feature_x'][:, :, :, :], (3, 0, 1, 2))  # data_format:'channels_last'
        test_set_y = group_value['test_onehot_y']
        class_number = len(set(group_value['train_int_y']))

        # layer parameters
        regularization = tf.keras.regularizers.L2(0.00001)
        initializer = tf.keras.initializers.HeNormal()
        # model structure
        inputs = tf.keras.Input(shape=(train_set_x.shape[1], train_set_x.shape[2], train_set_x.shape[3]))
        x = tf.keras.layers.Conv2D(32, 7, dilation_rate=1, strides=2, padding='same', data_format='channels_last')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
        x = tf.keras.layers.Conv2D(64, 5, dilation_rate=1, strides=2, padding='same', data_format='channels_last')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
        x = tf.keras.layers.Conv2D(128, 3, dilation_rate=1, strides=2, padding='same', data_format='channels_last')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
        x = tf.keras.layers.Flatten(data_format='channels_last')(x)
        x = tf.keras.layers.Dense(100, kernel_regularizer=regularization, kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(class_number, kernel_regularizer=regularization, kernel_initializer=initializer)(x)
        outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_cnn")
        # view models
        model.summary()

        # model parameters
        num_epochs = 10
        decay_epochs = 25
        batch_size = 512
        decay_steps = decay_epochs * len(train_set_y) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=decay_steps, decay_rate=0.3)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-08)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
        # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

        # train model
        now = datetime.datetime.now()
        model.fit(train_set_x, train_set_y, validation_split=0.1, epochs=num_epochs, batch_size=batch_size, shuffle=True, verbose='auto')
        print(datetime.datetime.now() - now)
        # test model
        predictions = model.predict(test_set_x)  # return predicted probabilities
        predict_y = np.argmax(predictions, axis=-1)  # return predicted labels
        test_loss, test_accuracy = model.evaluate(test_set_x, test_set_y)  # return loss and accuracy values

        results.append({"true_value": group_value['test_int_y'], "predict_softmax": predictions, "predict_value": predict_y,
            "predict_accuracy": test_accuracy})
        models.append(model)

        tf.keras.backend.clear_session()  #  clear the current session (Graph) and the model should be removed from GPU to release memory.
        del model

    return models, results


##
