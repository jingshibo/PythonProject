## import modules
import datetime
import numpy as np
import tensorflow as tf

## a "many to one" CnnGru model that returns only the last output
def classifyCnnGruLastOneModel(shuffled_groups):
    results = []
    for group_number, group_value in shuffled_groups.items():
        # input data
        train_set_x = group_value['train_feature_x'][:, :, :, :, 0:16]
        train_set_y = group_value['train_onehot_y'][:, 0, :]
        test_set_x = group_value['test_feature_x'][:, :, :, :, 0:16]
        test_set_y = group_value['test_onehot_y'][:, 0, :]
        class_number = len(set(group_value['train_int_y'][:, 0]))

        # layer parameters
        regularization = tf.keras.regularizers.L2(0.00001)
        initializer = tf.keras.initializers.HeNormal()

        # model structure
        # first, process each image with a cnn model
        cnn_inputs = tf.keras.Input(shape=(train_set_x.shape[2], train_set_x.shape[3], train_set_x.shape[4]))
        x = tf.keras.layers.Conv2D(32, 3, dilation_rate=1, padding='same', data_format='channels_last')(cnn_inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=1, padding="same")(x)
        cnn_outputs = tf.keras.layers.Flatten(data_format='channels_last')(x)
        cnn_model = tf.keras.Model(inputs=cnn_inputs, outputs=cnn_outputs, name="cnn_model")

        # then, process the features extracted from cnn model using a gru model
        gru_inputs = tf.keras.Input(shape=(train_set_x.shape[1], train_set_x.shape[2], train_set_x.shape[3], train_set_x.shape[4]))
        x = tf.keras.layers.TimeDistributed(cnn_model)(gru_inputs)  # call the cnn model built above at each timestep
        # x = tf.keras.layers.GRU(1000, return_sequences=True, dropout=0.1)(x)
        x = tf.keras.layers.GRU(1000, return_sequences=False, dropout=0.1)(x)
        x = tf.keras.layers.Dense(600, kernel_regularizer=regularization, kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(class_number)(x)
        gru_outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=gru_inputs, outputs=gru_outputs, name="gru_model")

        # view model
        model.summary()

        # model parameters
        num_epochs = 60
        decay_epochs = 25
        batch_size = 1024
        decay_steps = decay_epochs * len(train_set_y) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.003, decay_steps=decay_steps, decay_rate=0.3)
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

        results.append({"model": model, "true_value": group_value['test_int_y'][:, 0], "predict_softmax": predictions, "predict_value": predict_y,
            "predict_accuracy": test_accuracy})
    return results


## a "many to many" CnnGru model that returns the sequence output
def classifyCnnGruSequenceModel(shuffled_groups):
    results = []
    for group_number, group_value in shuffled_groups.items():
        # input data
        train_set_x = group_value['train_feature_x'][:, :, :, :, 0:16]
        train_set_y = group_value['train_onehot_y']
        test_set_x = group_value['test_feature_x'][:, :, :, :, 0:16]
        test_set_y = group_value['test_onehot_y']
        class_number = len(set(group_value['train_int_y'][:, 0]))

        # layer parameters
        regularization = tf.keras.regularizers.L2(0.00001)
        initializer = tf.keras.initializers.HeNormal()

        # model structure
        # first, process each image with a cnn model
        cnn_inputs = tf.keras.Input(shape=(train_set_x.shape[2], train_set_x.shape[3], train_set_x.shape[4]))
        x = tf.keras.layers.Conv2D(32, 5, dilation_rate=1, padding='same', data_format='channels_last')(cnn_inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=1, padding="same")(x)
        cnn_outputs = tf.keras.layers.Flatten(data_format='channels_last')(x)
        cnn_model = tf.keras.Model(inputs=cnn_inputs, outputs=cnn_outputs, name="cnn_model")

        # then, process the features extracted from cnn model using a gru model
        gru_inputs = tf.keras.Input(shape=(train_set_x.shape[1], train_set_x.shape[2], train_set_x.shape[3], train_set_x.shape[4]))
        x = tf.keras.layers.TimeDistributed(cnn_model)(gru_inputs)  # call the cnn model built above at each timestep
        # x = tf.keras.layers.GRU(1000, return_sequences=True, dropout=0.1)(x)
        x = tf.keras.layers.GRU(1000, return_sequences=True, dropout=0.1)(x)
        x = tf.keras.layers.Dense(600, kernel_regularizer=regularization, kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(class_number)(x)
        gru_outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=gru_inputs, outputs=gru_outputs, name="gru_model")

        # view model
        model.summary()

        # model parameters
        num_epochs = 50
        decay_epochs = 25
        batch_size = 1024
        decay_steps = decay_epochs * len(train_set_y) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.003, decay_steps=decay_steps, decay_rate=0.3)
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

        results.append({"model": model, "true_value": group_value['test_int_y'].flatten(), "predict_softmax": np.reshape(predictions,
            (-1, predictions.shape[-1])), "predict_value": predict_y.flatten(), "predict_accuracy": test_accuracy})
    return results