##
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import datetime
from collections import Counter
from sklearn.metrics import confusion_matrix
import json
import os

## select the input dataset
def selectInput(data_dict, indices):
    selected_data = {}
    for group, sets in data_dict.items():
        selected_data[group] = {}
        for set_type, mode_values in sets.items():
            selected_data[group][set_type] = {}
            for mode, list_of_windows in mode_values.items():
                # Selecting elements from each list based on indices
                selected_data[group][set_type][mode] = [[window_features[i] for i in indices if i < len(window_features)] for window_features in list_of_windows]
    return selected_data
# def select_elements(dict_of_lists, indices):
#     selected_data = {}
#     for key, nested_lists in dict_of_lists.items():
#         # Selecting elements based on indices and maintaining the nested list structure
#         selected_data[key] = [[lst[i] for i in indices if i < len(lst)] for lst in nested_lists]
#     return selected_data


## combine data of all gait events into a single dataset and normalize them
def combineNormalizedDataset(cross_validation_groups):
    normalized_groups = {}
    for group_number, group_value in cross_validation_groups.items():

        # initialize training set and test set for each group
        train_feature_x = []
        train_feature_y = []
        test_feature_x = []
        test_feature_y = []
        for set_type, set_value in group_value.items():
            if set_type == 'train_set':  # combine all data into a dataset
                for gait_event_label, gait_event_features in set_value.items():
                    train_feature_x.extend(np.array(gait_event_features))
                    train_feature_y.extend([gait_event_label] * len(gait_event_features))
            elif set_type == 'test_set':  # keep the structure unchanged
                for gait_event_label, gait_event_features in set_value.items():
                    test_feature_x.extend(np.array(gait_event_features))
                    test_feature_y.extend([gait_event_label] * len(gait_event_features))

        # normalization
        train_norm_x = (train_feature_x - np.mean(train_feature_x, axis=0)) / np.std(train_feature_x, axis=0)
        test_norm_x = (test_feature_x - np.mean(train_feature_x, axis=0)) / np.std(train_feature_x, axis=0)
        # one-hot encode categories (according to the alphabetical order)
        train_int_y = LabelEncoder().fit_transform(train_feature_y)
        train_onehot_y = tf.keras.utils.to_categorical(train_int_y)
        test_int_y = LabelEncoder().fit_transform(test_feature_y)
        test_onehot_y = tf.keras.utils.to_categorical(test_int_y)

        # put training data and test data into one group
        normalized_groups[group_number] = {"train_feature_x": train_norm_x, "train_int_y": train_int_y, "train_onehot_y": train_onehot_y,
            "test_feature_x": test_norm_x, "test_int_y": test_int_y, "test_onehot_y": test_onehot_y}
    return normalized_groups


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
        model.add(tf.keras.layers.InputLayer(input_shape=(train_set_x.shape[1])))  # or replaced by: model.add(tf.keras.Input(shape=1040))
        model.add(tf.keras.layers.Dense(200, kernel_regularizer=regularization))  # or activation=tf.nn.relu
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(200, kernel_regularizer=regularization))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(200, kernel_regularizer=regularization))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.Dense(100, kernel_regularizer=regularization, kernel_initializer=initializer))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.ReLU())
        # model.add(tf.keras.layers.Dropout(0.5))
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
    return models, results


## reorganize prediction results into separate lists in dicts
def reorganizePredictionResults(model_results):
    true_dict = []
    predict_dict = []  # save reorganized results for each group
    for result in model_results:
        true_value = result['true_value']
        predict_value = result['predict_value']
        # Find the indices where the value changes
        change_points = np.where(np.diff(true_value) != 0)[0] + 1
        true_list = np.split(true_value, change_points)
        predict_list = np.split(predict_value, change_points)
        # convert predict results into a dict
        true_dict.append({np.unique(true)[0]: true for true in true_list})
        predict_dict.append({np.unique(true)[0]: predict for true, predict in zip(true_list, predict_list)})
    return predict_dict, true_dict


## realization of the majority vote method with a window_size = 2 * n + 1
def majority_vote(predict_value, n):
    majority_classes = []

    for i in range(len(predict_value)):
        # Adjust the start and end of the window:  the window size varies at the beginning and the end of the array.
        start = max(0, i - n)  # At the start of the array, the window begins with fewer elements and expands to up to 2n+1 elements
        end = min(len(predict_value), i + n + 1)  # Towards the end of the array, the window size decreases as necessary
        # Extract the window
        window = predict_value[start:end]

        # Count occurrences and find the class with the majority
        counts = Counter(window)
        majority_class = counts.most_common(1)[0][0]
        majority_classes.append(majority_class)

    majority_vote_results = np.array(majority_classes)
    return majority_vote_results


## average classification accuracy and cm values
def calculateAccuracy(predict_mv_results, true_labels):
    accuracy_bygroup = []
    cm_bygroup = []
    for group_number, each_group in enumerate(predict_mv_results):
        combined_predict_labels = np.concatenate(list(each_group.values()))
        combined_true_labels = np.concatenate(list(true_labels[group_number].values()))

        numCorrect = np.count_nonzero(combined_true_labels == combined_predict_labels)
        locomotion_accuracy = numCorrect / len(combined_true_labels) * 100
        locomotion_cm = confusion_matrix(y_true=combined_true_labels, y_pred=combined_predict_labels)

        accuracy_bygroup.append(locomotion_accuracy)
        cm_bygroup.append(locomotion_cm)

    # calculate average values
    average_accuracy = np.mean(np.vstack(accuracy_bygroup))
    average_cm_number = np.mean(np.stack(cm_bygroup), axis=0)
    average_cm_recall = np.around(average_cm_number.astype('float') / average_cm_number.sum(axis=1)[:, np.newaxis], 3)  # calculate cm recall

    return average_accuracy, average_cm_number, average_cm_recall


## save classification accuracy and cm recall values
def saveResult(subject, average_accuracy, average_cm_number, average_cm_recall, model_type, result_set, project='Bipolar_Data'):
    data_dir = f'D:\Data\{project}\subject_{subject}\\results'
    result_file = f'subject_{subject}_model_{model_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    # Combine the two dictionaries into one
    combined_results = {'accuracy': average_accuracy, 'cm_num': average_cm_number.tolist(), 'cm_recall': average_cm_recall.tolist()}
    # Save to JSON file
    with open(result_path, 'w') as f:
        json.dump(combined_results, f, indent=8)


## read classification accuracy and cm recall values
def loadResult(subject, model_type, result_set, project='Bipolar_Data'):
    data_dir = f'D:\Data\{project}\subject_{subject}\\results'
    result_file = f'subject_{subject}_model_{model_type}_results_{result_set}.json'
    result_path = os.path.join(data_dir, result_file)

    with open(result_path, 'r') as f:
        loaded_data = json.load(f)

    combined_results = {'accuracy': loaded_data['accuracy'], 'cm_recall': np.array(loaded_data['cm_recall']),
        'cm_num': np.array(loaded_data['cm_num'])}
    return combined_results
