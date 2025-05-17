##  import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import numpy as np
import datetime
import os

# 2d CNN, kernel = (3,15), stride = (3, 15)


## design model
class Raw_Cnn_2d(nn.Module):
    def __init__(self, input_size, class_number):
        super(Raw_Cnn_2d, self).__init__()

        # define layer parameter
        self.conv1_parameter = [32, 3]
        self.conv2_parameter = [32, 3]
        self.conv3_parameter = [32, 3]
        # self.conv4_parameter = [32, 3]
        self.linear1_parameter = 500
        self.linear2_parameter = 100

        # define convolutional layer
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=self.conv1_parameter[0], kernel_size=self.conv1_parameter[1], dilation=2, stride=2),
            nn.BatchNorm2d(self.conv1_parameter[0]),
            nn.LeakyReLU(0.01), #nn.LeakyReLU(0.2)
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0),

            nn.Conv2d(in_channels=self.conv1_parameter[0], out_channels=self.conv2_parameter[0], kernel_size=self.conv2_parameter[1], dilation=2, stride=2),
            nn.BatchNorm2d(self.conv2_parameter[0]),
            nn.LeakyReLU(0.01),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0),

            nn.Conv2d(in_channels=self.conv2_parameter[0], out_channels=self.conv3_parameter[0], kernel_size=self.conv3_parameter[1], dilation=2, stride=2),
            nn.BatchNorm2d(self.conv3_parameter[0]),
            nn.LeakyReLU(0.01),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0),

            # nn.Conv2d(in_channels=self.conv3_parameter[0], out_channels=self.conv4_parameter[0], kernel_size=self.conv4_parameter[1], dilation=2, stride=1),
            # nn.BatchNorm2d(self.conv4_parameter[0]),
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        )

        # define dense layer
        self.linear_layer = nn.Sequential(
            nn.LazyLinear(self.linear1_parameter),
            nn.BatchNorm1d(self.linear1_parameter),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),

            nn.LazyLinear(self.linear2_parameter),
            nn.BatchNorm1d(self.linear2_parameter),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),

            nn.LazyLinear(class_number)
        )
        # self.initialize_weights()

    # define the initialization method for each layer
    def initialize_weights(self):
        for m in self.convolutional_layer:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)  # xavier / Glorot method
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                pass
            elif isinstance(m, nn.Linear):
                pass

    def forward(self, x, intermediate_features=False):
        x = self.convolutional_layer(x)
        cnn_features = x.detach().clone()
        x = torch.flatten(x, 1)
        flatten_features = x.detach().clone()
        x = self.linear_layer(x)
        # if self.training is False:
        #     x = F.softmax(x, dim=1)

        # obtain intermediate_features
        if intermediate_features:
            return x, cnn_features, flatten_features
        else:
            return x

# print("change dropout!")


## training
class ModelTraining():
    def __init__(self, num_epochs, batch_size, report_period=10):
        #  initialize member variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    def trainModel(self, classify_emg_dict, cross_validation_indices, decay_epochs, select_channels='emg_all'):
        models = []
        results = []
        # train and test the dataset for each group
        fold_number = len(cross_validation_indices)
        for fold_id in range(fold_number):
        # for group_number, group_value in {'group_1': shuffled_groups['group_1'], 'group_3': shuffled_groups['group_3']}.items():
            # initialize the tensorboard writer
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(os.path.join(self.result_dir, f'experiment_{timestamp}'))

            # extract the dataset
            group_number = f"group_{fold_id}"
            input_size = classify_emg_dict['data_x'].shape[1]  # input channel number
            class_number = len(set(classify_emg_dict['int_y']))
            data_set = self.selectSamples(classify_emg_dict, select_channels)

            # dataset of a fold
            self.train_loader, self.test_loader = foldDataloader(data_set, cross_validation_indices, fold_id, self.batch_size,
                onehot_label=False, shuffle_train=True, shuffle_test=False, drop_last=True)

            # training parameters
            self.model = Raw_Cnn_2d(input_size, class_number).to(self.device)  # move the model to GPU
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)  # initial learning rate and regularization
            self.loss_fn = torch.nn.CrossEntropyLoss()  # Loss functions expect data in batches
            decay_steps = decay_epochs * len(self.train_loader)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_steps, gamma=0.1)  # adjusted learning rate

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
    def __init__(self, data_x, data_y, indices):
        self.data_x = data_x
        self.labels = data_y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.data_x[i], self.labels[i]

## build a dataloader to load data from each fold
def foldDataloader(classify_emg_dict, cross_validation_indices, fold_id, batch_size, onehot_label=False, shuffle_train=True,
        shuffle_test=False, drop_last=True):
    fold = cross_validation_indices[fold_id]
    train_idx = fold['train']
    val_idx = fold['val']

    labels = classify_emg_dict['onehot_y'] if onehot_label else classify_emg_dict['int_y']
    data_x = classify_emg_dict['data_x']

    train_dataset = EmgDataSet(data_x, labels, train_idx)
    val_dataset = EmgDataSet(data_x, labels, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_test)

    return train_loader, val_loader


