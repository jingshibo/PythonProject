## import modules
import numpy as np
import tensorflow as tf
import datetime


##
# basic information
subject = "Shibo"
version = 1  # which experiment data to process
feature_set = 1  # which feature set to use

## load trained model
fold = 5
type = 1  # pre-trained model type
models = []
for number in range(fold):
    model_dir = f'D:\Data\Insole_Emg\subject_{subject}\Experiment_{version}\models_{type}\cross_validation_set_{number}'
    models.append(tf.keras.models.load_model(model_dir))

## view models
# class_number = 4
# model = models[2]
# model.summary()
# pretrained_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
# x = tf.keras.layers.Dense(class_number, name='dense_new')(pretrained_model.layers[-1].output)
# output = tf.keras.layers.Softmax()(x)
# transferred_model = tf.keras.Model(inputs=model.input, outputs=output)
# for layer in transferred_model.layers[:-2]:
#     layer.trainable = False  # all pretrained layers are frozen
# transferred_model.summary()


## transfer learning
group_results = []
now = datetime.datetime.now()
for number, (group_number, group_value) in enumerate(shuffled_groups.items()):
    predict_results = {}
    for transition_type, transition_train_data in group_value["train_set"].items():
        # training data
        train_set_x = transition_train_data['feature_x'][:, 0:1040]
        train_set_y = transition_train_data['feature_onehot_y']
        class_number = len(set(transition_train_data['feature_int_y']))

        # model
        pretrained_model = tf.keras.Model(inputs=models[number].input, outputs=models[number].layers[-3].output)
        x = tf.keras.layers.Dense(class_number, name='dense_new')(pretrained_model.layers[-1].output)
        output = tf.keras.layers.Softmax()(x)
        transferred_model = tf.keras.Model(inputs=models[number].input, outputs=output)
        for layer in transferred_model.layers[:-2]:
            layer.trainable = True  # all layers are trainable
        # view model
        transferred_model.summary()

        # model parameters
        num_epochs = 100
        decay_epochs = 30
        batch_size = 512
        decay_steps = decay_epochs * len(train_set_y) / batch_size
        # model configuration
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=decay_steps, decay_rate=0.3)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-08)
        transferred_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
        # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

        # train model
        transferred_model.fit(train_set_x, train_set_y, validation_split=0.1, epochs=num_epochs, batch_size=batch_size, shuffle=True, verbose='auto')

        # test data
        test_set_x = group_value['test_set'][transition_type]['feature_x'][:, 0:1040]
        test_set_y = group_value['test_set'][transition_type]['feature_onehot_y']
        # test model
        predictions = transferred_model.predict(test_set_x)  # return predicted probabilities
        predict_y = np.argmax(predictions, axis=-1)  # return predicted labels
        test_loss, test_accuracy = transferred_model.evaluate(test_set_x, test_set_y)  # return loss and accuracy values

        predict_results[transition_type] = {"model": transferred_model, "true_value": group_value['test_set'][transition_type]['feature_int_y'],
            "predict_value": predict_y, "predict_accuracy": test_accuracy}
    group_results.append(predict_results)
print(datetime.datetime.now() - now)