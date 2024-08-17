import os
import json
import numpy as np
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
import datetime


## a single CNN model
def classifyUsingCnnModel(shuffled_groups):
    '''
    A basic 2-layer CNN model
    '''

    models = []
    results = []

    for group_number, group_value in shuffled_groups.items():
        # input data
        train_set_x = group_value['train_feature_x']  # data_format:'channels_last'
        train_set_y = group_value['train_onehot_y']
        test_set_x = group_value['test_feature_x']  # data_format:'channels_last'
        test_set_y = group_value['test_onehot_y']
        class_number = len(set(group_value['train_int_y']))

        # layer parameters
        regularization = tf.keras.regularizers.L2(0.0001)
        initializer = tf.keras.initializers.HeNormal()
        # model structure
        inputs = tf.keras.Input(shape=(train_set_x.shape[1], train_set_x.shape[2], train_set_x.shape[3]))
        x = tf.keras.layers.Conv2D(30, 5, dilation_rate=2, data_format='channels_last')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
        x = tf.keras.layers.Flatten(data_format='channels_last')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(class_number, kernel_regularizer=regularization)(x)
        outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_cnn")
        # view models
        model.summary()

        # model parameters
        num_epochs = 60
        decay_epochs = 20
        batch_size = 512
        decay_steps = decay_epochs * len(train_set_y) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.005, decay_steps=decay_steps, decay_rate=0.3)
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

    return models, results


## save classification accuracy and cm recall values
def saveResult(subject, average_accuracies, average_cm_numbers, average_cm_recalls, model_type, result_set, project='HDsEMG_Recognition'):
    data_dir = f'D:\Data\{project}\subject_{subject}\\results'
    result_file = f'subject_{subject}_model_{model_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    # Combine the two dictionaries into one
    combined_results = {'accuracy': average_accuracies, 'cm_num': [arr.tolist() for arr in average_cm_numbers],
        'cm_recall': [arr.tolist() for arr in average_cm_recalls]}

    # Save to JSON file
    with open(result_path, 'w') as f:
        json.dump(combined_results, f, indent=8)


## read classification accuracy and cm recall values
def loadResult(subject, model_type, result_set, project='HDsEMG_Recognition'):
    data_dir = f'D:\Data\{project}\subject_{subject}\\results'
    result_file = f'subject_{subject}_model_{model_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    with open(result_path, 'r') as f:
        loaded_data = json.load(f)

    combined_results = {'accuracy': loaded_data['accuracy'], 'cm_recall': [np.array(lst) for lst in loaded_data['cm_recall']],
        'cm_num': [np.array(lst) for lst in loaded_data['cm_num']]}
    return combined_results
