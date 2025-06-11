##  import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import os
import gc


## training
class ModelTesting():
    def __init__(self, models, batch_size):
        #  initialize member variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_models = models
        self.batch_size = batch_size
        self.model = None
        self.train_loader = None
        self.test_loader = None

    #  train the model
    def testModel(self, classify_emg_dict, select_channels='emg_all'):
        results = []

        # test the dataset for each fold
        for fold_id, fold_data in enumerate(classify_emg_dict):

            # extract the dataset
            group_number = f"group_{fold_id}"
            input_channel = fold_data['X_train'].shape[1]  # input channel number
            class_number = len(set(fold_data['y_train_int']))
            data_set = self.selectSamples(fold_data, select_channels)

            # dataset of a fold
            self.train_loader, self.test_loader = foldDataloader(data_set, self.batch_size, onehot_label=False, shuffle_train=True,
                shuffle_test=False, drop_last=True)
            self.model = copy.deepcopy(self.pretrained_models[fold_id]).to(self.device)  # move the model to GPU

            # output the final test results
            test_true_labels, test_predict_softmax, test_predict_labels = self.predictTestResults(self.model, group_number)
            results.append(
                {"true_value": test_true_labels, "predict_softmax": test_predict_softmax, "predict_value": test_predict_labels})

            torch.cuda.empty_cache()
            gc.collect()
        return results

    #  classify test set
    def predictTestResults(self, model, group_number):
        model.train(False)

        # predict result statistics
        test_true_labels = []
        test_predict_softmax = []
        test_predict_labels = []

        with torch.no_grad():  # close autograd
            # prediction result statistics
            num_sample = 0  # total number
            num_correct = 0  # correct number

            # loop over each batch
            for i, test_data in enumerate(self.test_loader):
                test_inputs, test_labels = test_data[0].to(self.device), test_data[1].to(self.device)
                test_outputs = model(test_inputs)
                test_softmax = F.softmax(test_outputs, dim=1)

                # calculate test accuracy value summation from all previous batches
                _, predicted = torch.max(test_outputs.data, 1)  # use outputs.data to remove the grad of outputs variable
                num_correct += (predicted == test_labels).sum().item()
                num_sample += predicted.size(0)

                # combine predict results
                test_predict_labels.extend(predicted.cpu().numpy())
                test_predict_softmax.extend(test_softmax.cpu().numpy())
                test_true_labels.extend(test_labels.cpu().numpy())
            test_results = {"true_value": test_true_labels, "predict_softmax": test_predict_softmax,
                "predict_value": test_predict_labels}

            # calculate average training accuracy and loss for one group
            test_accuracy = num_correct / num_sample
            print(f"group: {int(group_number[-1])}, test accuracy: {test_accuracy:>7f}")
        return np.array(test_true_labels), np.array(test_predict_softmax), np.array(test_predict_labels)

    # select specific channels for model training and testing
    def selectSamples(self, group_value, select_channels='emg_all', bipolar_position=(0, 0)):
        # training dataset
        if select_channels == 'emg_all':
            return group_value
        elif select_channels == 'emg_1':
            data_x = group_value['train_feature_x'][:, :, :, 0: 65]
        elif select_channels == 'emg_2':
            data_x = group_value['train_feature_x'][:, :, :, 65: 130]
        elif select_channels == 'bipolar':
            pass
        else:
            raise Exception("No Such Channels")
        data_set = {'data_x': data_x, 'int_y': group_value['int_y'], 'onehot_y': group_value['onehot_y'],
            'label_map': group_value['label_map']}
        return data_set


## load data one by one from a fold
class EmgDataSet(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.labels = data_y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.labels[idx]

## build a dataloader to load data from each fold
def foldDataloader(fold_data, batch_size, onehot_label=False, shuffle_train=True, shuffle_test=False, drop_last=True):
    X_train = fold_data['X_train']
    y_train = fold_data['y_train_onehot'] if onehot_label else fold_data['y_train_int']
    X_test = fold_data['X_test']
    y_test = fold_data['y_test_onehot'] if onehot_label else fold_data['y_test_int']

    train_dataset = EmgDataSet(X_train, y_train)
    test_dataset = EmgDataSet(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

    return train_loader, test_loader

