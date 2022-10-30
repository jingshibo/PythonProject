## import modules
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from Models.Utility_Functions import Confusion_Matrix


## training model

def classifyUsingAnnModel(shuffled_groups):
    '''
    A basic 4 layer ANN model
    '''
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
        #     emg_feature_bipolar_x = np.concatenate((emg_feature_bipolar_x, group_value[:, 33+65*i].reshape(len(emg_feature_y), 1)), axis=1)
        # train_set_x = train_rectus_x[:, 0:520]
        # train_set_y = group_value['train_onehot_y']
        # test_set_x = test_rectus_x[:, 0:520]
        # test_set_y = group_value['test_onehot_y']

        # input data
        train_set_x = group_value['train_feature_x'][:, 0:1040]
        train_set_y = group_value['train_onehot_y']
        test_set_x = group_value['test_feature_x'][:, 0:1040]
        test_set_y = group_value['test_onehot_y']
        class_number = len(set(group_value['train_int_y']))

        # layer parameters
        regularization = tf.keras.regularizers.L2(0.00001)
        initializer = tf.keras.initializers.HeNormal()
        # model structure
        model = tf.keras.models.Sequential(name="ann_model")  # optional name
        model.add(tf.keras.layers.InputLayer(input_shape=(train_set_x.shape[1]))) # It can also be replaced by: model.add(tf.keras.Input(shape=(28,28)))
        model.add(tf.keras.layers.Dense(600, kernel_regularizer=regularization, kernel_initializer=initializer))  # or activation=tf.nn.relu
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(600, kernel_regularizer=regularization, kernel_initializer=initializer))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(600, kernel_regularizer=regularization, kernel_initializer=initializer))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(100, kernel_regularizer=regularization, kernel_initializer=initializer))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(class_number))
        model.add(tf.keras.layers.Softmax())  # or activation=tf.nn.softmax
        # view model
        model.summary()


        # model parameters
        num_epochs = 100
        decay_epochs = 30
        batch_size = 1024
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

        results.append({"model": model, "true_value": group_value['test_int_y'], "predict_value": predict_y, "predict_accuracy": test_accuracy})
    return results


## majority vote
def majorityVoteResults(classify_results, window_per_repetition):
    '''
    The majority vote results for each transition repetition
    '''
    bin_results = []
    for result in classify_results:
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
    for result in bin_results:
        true_y = []
        predict_y = []
        for key, value in result.items():
            if key == 'true_value':
                true_y = [np.bincount(i).argmax() for i in value]
            elif key == 'predict_value':
                predict_y = [np.bincount(i).argmax() for i in value]
        majority_results.append({"true_value": np.array(true_y), "predict_value": np.array(predict_y)})
    return majority_results


## accuracy
def averageAccuracy(majority_results):
    '''
    The accuracy for each cross validation group and average value across groups
    '''
    cm = []
    accuracy = []
    for result in majority_results:
        true_y = result['true_value']
        predict_y = result['predict_value']
        num_Correct = np.count_nonzero(true_y == predict_y)
        accuracy.append(num_Correct / len(true_y) * 100)
        cm.append(confusion_matrix(y_true=true_y, y_pred=predict_y))
    mean_accuracy = sum(accuracy) / len(accuracy)
    sum_cm = np.sum(np.array(cm), axis=0)
    return mean_accuracy, sum_cm


## plot confusion matrix
def confusionMatrix(sum_cm, recall=False):
    '''
    plot overall confusion matrix recall values
    '''
    plt.figure()
    # the label order in the classes list should correspond to the one hot labels, which is a alphabetical order
    # class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'LW', 'SALW', 'SASA', 'SASS', 'SA', 'SDLW', 'SDSD', 'SDSS', 'SD', 'SSLW', 'SSSA', 'SSSD']
    class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'SALW', 'SASA', 'SASS', 'SDLW', 'SDSD', 'SDSS', 'SSLW', 'SSSA', 'SSSD']
    cm_recall = Confusion_Matrix.plotConfusionMatrix(sum_cm, class_labels, normalize=recall)
    return cm_recall


