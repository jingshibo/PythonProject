## import modules
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from Models.Utility_Functions import Confusion_Matrix
from scipy.linalg import block_diag

##
def classifyMultipleAnnModel(shuffled_groups):
    '''
    train multiple 4-layer ANN models for each group of dataset (grouped based on prior information)
    '''
    group_results = []
    for group_number, group_value in shuffled_groups.items():
        predict_results = {}
        for transition_type, transition_train_data in group_value["train_set"].items():
            # training data
            train_set_x = transition_train_data['feature_x'][:, 0:1040]
            train_set_y = transition_train_data['feature_onehot_y']
            class_number = len(set(transition_train_data['feature_int_y']))

            # layer parameters
            regularization = tf.keras.regularizers.L2(0.00001)
            initializer = tf.keras.initializers.HeNormal()
            # model structure
            model = tf.keras.models.Sequential(name="ann_model")  # optional name
            model.add(tf.keras.layers.InputLayer(input_shape=(train_set_x.shape[1])))  # or replaced by: model.add(tf.keras.Input(shape=(28,28)))
            model.add(tf.keras.layers.Dense(200, kernel_regularizer=regularization, kernel_initializer=initializer))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(200, kernel_regularizer=regularization, kernel_initializer=initializer))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(200, kernel_regularizer=regularization, kernel_initializer=initializer))
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

            # training model
            now = datetime.datetime.now()
            model.fit(train_set_x, train_set_y, validation_split=0.1, epochs=num_epochs, batch_size=batch_size, shuffle=True, verbose='auto')
            print(datetime.datetime.now() - now)

            # test data
            test_set_x = group_value['test_set'][transition_type]['feature_x'][:, 0:1040]
            test_set_y = group_value['test_set'][transition_type]['feature_onehot_y']
            # test model
            predictions = model.predict(test_set_x)  # return predicted probabilities
            predict_y = np.argmax(predictions, axis=-1)  # return predicted labels
            test_loss, accuracy = model.evaluate(test_set_x, test_set_y)  # return loss and accuracy values

            predict_results[transition_type] = {"model": model, "true_value": group_value['test_set'][transition_type]['feature_int_y'],
                "predict_value": predict_y, "predict_accuracy": accuracy}
        group_results.append(predict_results)
    return group_results


## majority vote
def majorityVoteResults(group_results, window_per_repetition):
    # reunite the samples belonging to the same transition
    bin_results = []
    for each_group in group_results:
        bin_transitions = {}
        for transition_type, transition_results in each_group.items():
            true_y = []
            predict_y = []
            for key, value in transition_results.items():
                if key == 'true_value':
                    for i in range(0, len(value), window_per_repetition):
                        true_y.append(value[i: i+window_per_repetition])
                elif key == 'predict_value':
                    for i in range(0, len(value), window_per_repetition):
                        predict_y.append(value[i: i+window_per_repetition])
            bin_transitions[transition_type] = {"true_value": true_y, "predict_value": predict_y}
        bin_results.append(bin_transitions)

    # use majority vote to get a consensus result for each repetition
    majority_results = []
    for each_group in bin_results:
        majority_transitions = {}
        for transition_type, transition_results in each_group.items():
            true_y = []
            predict_y = []
            for key, value in transition_results.items():
                if key == 'true_value':
                    true_y = [np.bincount(i).argmax() for i in value]
                elif key == 'predict_value':
                    predict_y = [np.bincount(i).argmax() for i in value]
            majority_transitions[transition_type] = {"true_value": np.array(true_y), "predict_value": np.array(predict_y)}
        majority_results.append(majority_transitions)
    return majority_results

# accuracy
def getAccuracyPerGroup(majority_results):
    accuracy = []
    cm = []
    for each_group in majority_results:
        transition_cm = {}
        transition_accuracy = {}
        for transition_type, transition_result in each_group.items():
            true_y = transition_result['true_value']
            predict_y = transition_result['predict_value']
            numCorrect = np.count_nonzero(true_y == predict_y)
            transition_accuracy[transition_type] = numCorrect / len(true_y) * 100
            transition_cm[transition_type] = confusion_matrix(y_true=true_y, y_pred=predict_y)
        accuracy.append(transition_accuracy)
        cm.append(transition_cm)
    return accuracy, cm

## average value
def averageAccuracy(accuracy, cm):
    transition_groups = list(accuracy[0].keys())  # list all transition types
    # average accuracy across groups
    average_accuracy = {transition: 0 for transition in transition_groups}  # initialize average accuracy list
    for group_values in accuracy:
        for transition_type, transition_accuracy in group_values.items():
            average_accuracy[transition_type] = average_accuracy[transition_type] + transition_accuracy
    for transition_type, transition_accuracy in average_accuracy.items():
        average_accuracy[transition_type] = transition_accuracy / len(accuracy)
    # overall cm across groups
    sum_cm = {transition: 0 for transition in transition_groups}   # initialize overall cm list
    for group_values in cm:
        for transition_type, transition_cm in group_values.items():
            sum_cm[transition_type] = sum_cm[transition_type] + transition_cm
    return average_accuracy, sum_cm

    ## plot confusion matrix
def confusionMatrix(sum_cm, recall=False):
    '''
    plot overall confusion matrix recall values
    '''
    # the label order in the classes list should correspond to the one hot labels, which is a alphabetical order
    list_cm = [cm for label, cm in sum_cm.items()]
    overall_cm = block_diag(*list_cm)  # create a diagonal matrix from multiple arrays.
    plt.figure()
    class_labels = ['LWLW', 'LWSA', 'LWSD', 'LWSS', 'SALW', 'SASA', 'SASS', 'SDLW', 'SDSD', 'SDSS', 'SSLW', 'SSSA', 'SSSD']
    cm_recall = Confusion_Matrix.plotConfusionMatrix(overall_cm, class_labels, normalize=recall)
    return cm_recall










    # mean_accuracy = {'transition_LW': 0, 'transition_SA': 0, 'transition_SD': 0, 'transition_SS': 0}
    # for value in accuracy:
    #     mean_accuracy['transition_LW'] = mean_accuracy['transition_LW'] + value['transition_LW']
    #     mean_accuracy['transition_SA'] = mean_accuracy['transition_SA'] + value['transition_SA']
    #     mean_accuracy['transition_SD'] = mean_accuracy['transition_SD'] + value['transition_SD']
    #     mean_accuracy['transition_SS'] = mean_accuracy['transition_SS'] + value['transition_SS']
    # for key, value in mean_accuracy.items():
    #     mean_accuracy[key] = value / len(accuracy)
    #
    # cm_recall = {'transition_LW': np.zeros((4, 4)), 'transition_SA': np.zeros((3, 3)), 'transition_SD': np.zeros((3, 3)),
    #     'transition_SS': np.zeros((3, 3))}
    # for value in cm:
    #     cm_recall['transition_LW'] = cm_recall['transition_LW'] + value['transition_LW']
    #     cm_recall['transition_SA'] = cm_recall['transition_SA'] + value['transition_SA']
    #     cm_recall['transition_SD'] = cm_recall['transition_SD'] + value['transition_SD']
    #     cm_recall['transition_SS'] = cm_recall['transition_SS'] + value['transition_SS']
    # for key, value in cm_recall.items():
    #     cm_recall[key] = (np.around(value.T / np.sum(value, axis=1) * 100, decimals=1)).T
    # return mean_accuracy, cm_recall