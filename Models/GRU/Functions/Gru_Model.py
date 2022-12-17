## import modules
import datetime
import numpy as np
import tensorflow as tf
from Models.Utility_Functions import Channel_Selection


## a "many to one" GRU model that returns only the last output
def classifyGtuLastOneModel(shuffled_groups):
    results = []
    for group_number, group_value in shuffled_groups.items():

        # select channels to calculate
        channel_to_compute = 'emg_all'
        train_set_x, train_set_y, test_set_x, test_set_y = Channel_Selection.select1dFeatureChannels(group_value, channel_to_compute)
        class_number = len(set(group_value['train_int_y'][:, 0]))

        # layer parameters
        regularization = tf.keras.regularizers.L2(0.00001)
        initializer = tf.keras.initializers.HeNormal()

        # model structure
        model = tf.keras.models.Sequential(name="gru_model")  # optional name
        model.add(tf.keras.layers.GRU(1000, return_sequences=True, dropout=0.5, input_shape=(None, train_set_x.shape[2])))
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
        decay_steps = decay_epochs * len(train_set_y) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=decay_steps, decay_rate=0.5)
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


## multiple "many to one" GRU model models by group
def classifyMultipleGruLastOneModel(shuffled_groups):
    group_results = []
    for group_number, group_value in shuffled_groups.items():
        predict_results = {}
        for transition_type, transition_train_data in group_value["train_set"].items():
            # training data
            train_set_x = transition_train_data['feature_norm_x'][:, :, 0:1040]
            train_set_y = transition_train_data['feature_onehot_y'][:, 0, :]
            class_number = len(set(transition_train_data['feature_int_y'][:, 0]))

            # layer parameters
            regularization = tf.keras.regularizers.L2(0.00001)
            initializer = tf.keras.initializers.HeNormal()
            # model structure
            model = tf.keras.models.Sequential(name="gru_model")  # optional name
            model.add(
                tf.keras.layers.GRU(1000, return_sequences=True, dropout=0, input_shape=(train_set_x.shape[1], train_set_x.shape[2])))
            model.add(tf.keras.layers.GRU(1000, return_sequences=False, dropout=0))
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
            decay_epochs = 20
            batch_size = 512
            decay_steps = decay_epochs * len(train_set_y) / batch_size
            # model configuration
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=decay_steps, decay_rate=0.5)
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-08)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
            # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

            # training model
            now = datetime.datetime.now()
            model.fit(train_set_x, train_set_y, validation_split=0.1, epochs=num_epochs, batch_size=batch_size, shuffle=True, verbose='auto')
            print(datetime.datetime.now() - now)

            # test data
            test_set_x = group_value['test_set'][transition_type]['feature_norm_x'][:, :, 0:1040]
            test_set_y = group_value['test_set'][transition_type]['feature_onehot_y'][:, 0, :]
            # test model
            predictions = model.predict(test_set_x)  # return predicted probabilities
            predict_y = np.argmax(predictions, axis=-1)  # return predicted labels
            test_loss, accuracy = model.evaluate(test_set_x, test_set_y)  # return loss and accuracy values

            predict_results[transition_type] = {"model": model, "true_value": group_value['test_set'][transition_type]['feature_int_y'][:, 0],
                "predict_softmax": predictions, "predict_value": predict_y, "predict_accuracy": accuracy}
        group_results.append(predict_results)
    return group_results


## a bidirectional "many to one" GRU model that returns only the last output
def classifyBiGtuLastOneModel(shuffled_groups):
    results = []
    for group_number, group_value in shuffled_groups.items():
        # input data
        train_set_x = group_value['train_feature_x'][:, :, 0:1040]
        train_set_y = group_value['train_onehot_y'][:, 0, :]
        test_set_x = group_value['test_feature_x'][:, :, 0:1040]
        test_set_y = group_value['test_onehot_y'][:, 0, :]
        class_number = len(set(group_value['train_int_y'][:, 0]))

        # layer parameters
        regularization = tf.keras.regularizers.L2(0.00001)
        initializer = tf.keras.initializers.HeNormal()

        # model structure
        model = tf.keras.models.Sequential(name="gru_model")  # optional name
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(1000, return_sequences=True, dropout=0.5, input_shape=(train_set_x.shape[1], train_set_x.shape[2]))))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1000, return_sequences=False, dropout=0.5)))
        model.add(tf.keras.layers.Dense(600, kernel_regularizer=regularization, kernel_initializer=initializer))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(class_number))
        model.add(tf.keras.layers.Softmax())  # or activation=tf.nn.softmax
        # view model
        # model.summary()


        # model parameters
        num_epochs = 50
        decay_epochs = 30
        batch_size = 1024
        decay_steps = decay_epochs * len(train_set_y) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=decay_steps, decay_rate=0.5)
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



## a "many to many" GRU model that returns an output sequence
def classifyGtuSequenceModel(shuffled_groups):
    results = []
    for group_number, group_value in shuffled_groups.items():
        # input data
        train_set_x = group_value['train_feature_x'][:, :, 0:1040]
        train_set_y = group_value['train_onehot_y']
        test_set_x = group_value['test_feature_x'][:, :, 0:1040]
        test_set_y = group_value['test_onehot_y']
        class_number = len(set(group_value['train_int_y'][:, 0]))

        # layer parameters
        regularization = tf.keras.regularizers.L2(0.00001)
        initializer = tf.keras.initializers.HeNormal()

        # model structure
        model = tf.keras.models.Sequential(name="gru_model")  # optional name
        model.add(tf.keras.layers.GRU(1000, return_sequences=True, dropout=0, input_shape=(train_set_x.shape[1], train_set_x.shape[2])))
        model.add(tf.keras.layers.GRU(1000, return_sequences=True, dropout=0))
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
        decay_steps = decay_epochs * len(train_set_y) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=decay_steps, decay_rate=0.5)
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

        # reorganize the data structure to fit the postprocess (majority vote functions) later
        results.append({"model": model, "true_value": group_value['test_int_y'].flatten(), "predict_softmax": np.reshape(predictions,
            (-1, predictions.shape[-1])), "predict_value": predict_y.flatten(), "predict_accuracy": test_accuracy})
    return results


## multiple "many to many" GRU model models by group
def classifyMultipleGruSequenceModel(shuffled_groups):
    group_results = []
    for group_number, group_value in shuffled_groups.items():
        predict_results = {}
        for transition_type, transition_train_data in group_value["train_set"].items():
            # training data
            train_set_x = transition_train_data['feature_norm_x'][:, :, 0:1040]
            train_set_y = transition_train_data['feature_onehot_y']
            class_number = len(set(transition_train_data['feature_int_y'][:, 0]))

            # layer parameters
            regularization = tf.keras.regularizers.L2(0.00001)
            initializer = tf.keras.initializers.HeNormal()
            # model structure
            model = tf.keras.models.Sequential(name="gru_model")  # optional name
            model.add(
                tf.keras.layers.GRU(1000, return_sequences=True, dropout=0, input_shape=(train_set_x.shape[1], train_set_x.shape[2])))
            model.add(tf.keras.layers.GRU(1000, return_sequences=True, dropout=0))
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
            decay_epochs = 20
            batch_size = 512
            decay_steps = decay_epochs * len(train_set_y) / batch_size
            # model configuration
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=decay_steps, decay_rate=0.5)
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-08)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
            # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

            # training model
            now = datetime.datetime.now()
            model.fit(train_set_x, train_set_y, validation_split=0.1, epochs=num_epochs, batch_size=batch_size, shuffle=True, verbose='auto')
            print(datetime.datetime.now() - now)

            # test data
            test_set_x = group_value['test_set'][transition_type]['feature_norm_x'][:, :, 0:1040]
            test_set_y = group_value['test_set'][transition_type]['feature_onehot_y']
            # test model
            predictions = model.predict(test_set_x)  # return predicted probabilities
            predict_y = np.argmax(predictions, axis=-1)  # return predicted labels
            test_loss, accuracy = model.evaluate(test_set_x, test_set_y)  # return loss and accuracy values

            predict_results[transition_type] = {"model": model, "true_value": group_value['test_set'][transition_type]['feature_int_y'].flatten(),
                "predict_softmax": np.reshape(predictions, (-1, predictions.shape[-1])), "predict_value": predict_y.flatten(), "predict_accuracy": accuracy}
        group_results.append(predict_results)
    return group_results



