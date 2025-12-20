##
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import datetime


## a single ANN model
def classifyUsingAnnModel(shuffled_groups):
    '''
    A basic 4-layer ANN model
    '''

    models = []
    results = []

    for group_number, group_value in shuffled_groups.items():

        # input data
        train_set_x = group_value['train_feature_x']
        train_set_y = group_value['train_onehot_y']
        test_set_x = group_value['test_feature_x']
        test_set_y = group_value['test_onehot_y']
        class_number = len(set(group_value['train_int_y']))

        # layer parameters
        regularization = tf.keras.regularizers.L2(0.0001)
        initializer = tf.keras.initializers.HeNormal()
        # model structure
        model = tf.keras.models.Sequential(name="ann_model")  # optional name
        model.add(tf.keras.layers.InputLayer(input_shape=(train_set_x.shape[1],)))  # or replaced by: model.add(tf.keras.Input(shape=1040))
        model.add(tf.keras.layers.Dense(100, kernel_regularizer=regularization))  # or activation=tf.nn.relu
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.Dense(200, kernel_regularizer=regularization))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.Dense(200, kernel_regularizer=regularization))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.Dense(100, kernel_regularizer=regularization, kernel_initializer=initializer))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(class_number))
        model.add(tf.keras.layers.Softmax())  # or activation=tf.nn.softmax
        # view model
        model.summary()

        # model parameters
        num_epochs = 20
        decay_epochs = 10
        batch_size = 128
        decay_steps = decay_epochs * len(train_set_y) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=decay_steps, decay_rate=0.5)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-08)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
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
    return models, results
