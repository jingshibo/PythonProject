##  import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import numpy as np
import datetime
import copy
import os
import gc


## training
class ModelTraining():
    def __init__(self, models, num_epochs, batch_size, report_period=10):
        #  initialize member variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_models = models
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.report_period = report_period
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.lr_scheduler = None
        self.train_loader = None
        self.test_loader = None
        self.writer = None
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = f'D:\Project\pythonProject\Model_Raw\CNN_2D\Results\\runs_{timestamp}'

    #  train the model
    def trainModel(self, classify_emg_dict, decay_epochs, select_channels='emg_all'):
        models = []
        results = []

        # train and test the dataset for each fold
        for fold_id, fold_data in enumerate(classify_emg_dict):
            # initialize the tensorboard writer
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            # self.writer = SummaryWriter(os.path.join(self.result_dir, f'experiment_{timestamp}'))

            # extract the dataset
            group_number = f"group_{fold_id}"
            input_channel = fold_data['X_train'].shape[1]  # input channel number
            class_number = len(set(fold_data['y_train_int']))
            data_set = self.selectSamples(fold_data, select_channels)

            # dataset of a fold
            self.train_loader, self.test_loader = foldDataloader(data_set, self.batch_size, onehot_label=False, shuffle_train=True,
                shuffle_test=False, drop_last=True)

            # training parameters
            self.model = copy.deepcopy(self.pretrained_models[fold_id]).to(self.device)  # move the model to GPU
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.00001)  # initial learning rate and regularization
            self.loss_fn = torch.nn.CrossEntropyLoss()  # Loss functions expect data in batches
            decay_steps = decay_epochs * len(self.train_loader)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_steps, gamma=0.3)  # adjusted learning rate

            # train and test the model of a fold
            for epoch_number in range(self.num_epochs):  # loop over each epoch
                self.trainOneEpoch(group_number, epoch_number)
                # classify the test set every 10 epochs
                if (epoch_number + 1) % self.report_period == 0:
                    test_true_labels, test_predict_softmax, test_predict_labels = self.predictTestResults(group_number, epoch_number)
            # log the model structure
            # self.writer.add_graph(self.model, next(iter(self.train_loader))[0].to(self.device))

            # output the final test results
            test_true_labels, test_predict_softmax, test_predict_labels = self.predictTestResults(group_number, self.num_epochs-1)
            results.append(
                {"true_value": test_true_labels, "predict_softmax": test_predict_softmax, "predict_value": test_predict_labels})
            models.append(self.model.to("cpu"))

            torch.cuda.empty_cache()
            gc.collect()
        return models, results

    #  conduct training of one epoch
    def trainOneEpoch(self, group_number, epoch_number):
        self.model.train(True)
        # loop over each batch
        for batch_number, data in enumerate(self.train_loader):
            inputs, labels = data[0].to(self.device), data[1].to(device=self.device, dtype=torch.long)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # output size: (batch_size, class_number)
            train_loss = self.loss_fn(outputs, labels)
            train_loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()  # update the learning rate

        # report the training results every 10 epochs
        if (epoch_number + 1) % self.report_period == 0:
            # calculate training accurate value and loss from only the last batches of one epoch
            _, predicted = torch.max(outputs.data, 1)  # use outputs.data to remove the grad of outputs variable
            num_correct = (predicted == labels).sum().item()
            num_sample = predicted.size(0)
            training_accuracy = num_correct / num_sample

            print(f"group: {int(group_number[-1])}, epoch: {epoch_number + 1}, train accuracy: {training_accuracy:>7f}, train loss: {train_loss.item():>7f}")
            # Log the average training accuracy and loss per epoch
            # self.writer.add_scalars('Accuracy', {f'{group_number}_train accuracy': training_accuracy}, epoch_number)
            # self.writer.add_scalars('Loss', {f'{group_number}_train loss': train_loss}, epoch_number )
            # self.writer.flush()
        else:
            print(f"group: {int(group_number[-1])}, epoch: {epoch_number + 1}, batch: {len(self.train_loader) * (epoch_number + 1)}, "
                f"learning rate: {self.lr_scheduler.get_last_lr()}")

    #  classify test set
    def predictTestResults(self, group_number, epoch_number):
        self.model.train(False)

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
                test_inputs, test_labels = test_data[0].to(self.device), test_data[1].to(device=self.device, dtype=torch.long)
                test_outputs = self.model(test_inputs)
                test_loss = self.loss_fn(test_outputs, test_labels)
                test_softmax = F.softmax(test_outputs, dim=1)

                # calculate test accurate value summation from all previous batches
                _, predicted = torch.max(test_outputs.data, 1)  # use outputs.data to remove the grad of outputs variable
                num_correct += (predicted == test_labels).sum().item()
                num_sample += predicted.size(0)

                # combine predict results
                test_predict_labels.extend(predicted.cpu().numpy())
                test_predict_softmax.extend(test_softmax.cpu().numpy())
                test_true_labels.extend(test_labels.cpu().numpy())
            test_results = {"true_value": test_true_labels, "predict_softmax": test_predict_softmax, "predict_value": test_predict_labels}

            # calculate average training accuracy and loss for one group
            test_accuracy = num_correct / num_sample
            print(f"group: {int(group_number[-1])}, epoch: {epoch_number + 1}, test accuracy: {test_accuracy:>7f}, test loss: {test_loss.item():>7f}")

            # Log the average test accuracy per epoch
            # self.writer.add_scalars('Accuracy', {f'{group_number}_test accuracy': test_accuracy}, epoch_number)
            # self.writer.add_scalars('Loss', {f'{group_number}_test loss': test_loss}, epoch_number)
            # self.writer.flush()

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


