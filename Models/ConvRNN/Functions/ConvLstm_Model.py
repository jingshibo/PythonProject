## import modules
import datetime
import numpy as np
import tensorflow as tf

## a "many to one" ConvRNN model that returns only the last output
def classifyConvLstmLastOneModel(shuffled_groups):
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
        inputs = tf.keras.Input(shape=(train_set_x.shape[1], train_set_x.shape[2], train_set_x.shape[3], train_set_x.shape[4]))
        x = tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), dropout=0.5, padding='same', dilation_rate=1, return_sequences=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=1, padding="same")(x)

        x = tf.keras.layers.Flatten(data_format='channels_last')(x)
        x = tf.keras.layers.Dense(1000, kernel_regularizer=regularization, kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(class_number, kernel_regularizer=regularization, kernel_initializer=initializer)(x)
        outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_convlstm")

        # view model
        model.summary()

        # model parameters
        num_epochs = 50
        decay_epochs = 25
        batch_size = 1024
        decay_steps = decay_epochs * len(train_set_y) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=decay_steps, decay_rate=0.3)
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


## a "many to many" ConvRNN model that returns only the sequence
def classifyConvLstmSequenceModel(shuffled_groups):
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
        inputs = tf.keras.Input(shape=(train_set_x.shape[1], train_set_x.shape[2], train_set_x.shape[3], train_set_x.shape[4]))
        x = tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), dropout=0.1, padding='same', dilation_rate=1, return_sequences=True)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(pool_size=2, strides=1, padding="same"))(x)

        x = tf.keras.layers.Reshape((train_set_x.shape[1], -1))(x)
        x = tf.keras.layers.Dense(1000, kernel_regularizer=regularization, kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(class_number, kernel_regularizer=regularization, kernel_initializer=initializer)(x)
        outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_convlstm")

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