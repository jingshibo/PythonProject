##  import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from torchinfo import summary


## design model
class Raw_Cnn_2d(nn.Module):
    def __init__(self, input_size, class_number):
        super(Raw_Cnn_2d, self).__init__()

        # define layer parameter
        self.conv1_parameter = [32, 7]
        self.conv2_parameter = [64, 5]
        self.conv3_parameter = [128, 3]
        self.linear1_parameter = 1000
        self.linear2_parameter = 100

        # define convolutional layer
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=self.conv1_parameter[0], kernel_size=self.conv1_parameter[1], stride=2),
            nn.BatchNorm2d(self.conv1_parameter[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=self.conv1_parameter[0], out_channels=self.conv2_parameter[0], kernel_size=self.conv2_parameter[1], stride=2),
            nn.BatchNorm2d(self.conv2_parameter[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=self.conv2_parameter[0], out_channels=self.conv3_parameter[0], kernel_size=self.conv3_parameter[1], stride=2),
            nn.BatchNorm2d(self.conv3_parameter[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        # define dense layer
        self.linear_layer = nn.Sequential(
            nn.LazyLinear(self.linear1_parameter),
            nn.BatchNorm1d(self.linear1_parameter),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.LazyLinear(self.linear2_parameter),
            nn.BatchNorm1d(self.linear2_parameter),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.LazyLinear(class_number)
        )

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        if self.training is False:
            x = F.softmax(x, dim=1)
        return x


## model summary
model = Raw_Cnn_2d(1, 13).to('cuda')  # move the model to GPU
summary(model, input_size=(1024, 1, 512, 130))


##  loading dataset
class EmgDataSet(Dataset):
    def __init__(self, shuffled_data, mode):
        self.n_samples = shuffled_data[f'{mode}_int_y'].shape[0]
        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(np.transpose(shuffled_data[f'{mode}_feature_x'], (3, 2, 0, 1)))  # size [n_samples, n_channel, length, width]
        self.y_data = torch.from_numpy(shuffled_data[f'{mode}_int_y'])  # size [n_samples]

    def __getitem__(self, index):   # support indexing such that dataset[i] can be used to get i-th sample
        return self.x_data[index, :, :, :], self.y_data[index]

    def __len__(self):  # we can call len(dataset) to return the size
        return self.n_samples


## training
class ModelTraining():
    def __init__(self, num_epochs, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.test_loader = None

    #  conduct training of one epoch
    def trainOneFold(self, group_number):
        self.model.train(True)
        # repeat training for epochs times
        for epoch_number in range(self.num_epochs):
            # loop over for one epoch
            for i, data in enumerate(self.train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f"group: {group_number[-1]}, epoch: {epoch_number}, batch: {i}, loss: {loss:>7f}")

    #  classify test set of one epoch
    def predictTestResults(self):
        self.model.train(False)
        test_true_labels = []
        test_predict_softmax = []
        test_predict_labels = []
        with torch.no_grad():  # close autograd
            for i, test_data in enumerate(self.test_loader):
                test_inputs, test_labels = test_data[0].to(self.device), test_data[1].to(self.device)
                predict_softmax = self.model(test_inputs)
                predict_label = np.argmax(predict_softmax.cpu().numpy(), axis=-1)  # return predicted labels
                test_true_labels.extend(test_labels.cpu().numpy())
                test_predict_softmax.extend(predict_softmax.cpu().numpy())
                test_predict_labels.extend(predict_label)
        return np.array(test_true_labels), np.array(test_predict_softmax), np.array(test_predict_labels)

    # select specific channels for model training and testing
    def selectSamples(self, group_value, select_channels, bipolar_position=(0, 0)):
        # training dataset
        if select_channels == 'emg_all':
            return group_value
        elif select_channels == 'emg_1':
            train_set_x = group_value['train_feature_x'][:, 0: 65, :, :]
            test_set_x = group_value['train_feature_x'][:, 0: 65, :, :]
        elif select_channels == 'emg_2':
            train_set_x = group_value['train_feature_x'][:, 65: 130, :, :]
            test_set_x = group_value['train_feature_x'][:, 65: 130, :, :]
        elif select_channels == 'bipolar':
            pass
        else:
            raise Exception("No Such Channels")
        data_set = {'train_feature_x': train_set_x, 'train_int_y': group_value['train_int_y'],
            'train_onehot_y': group_value['train_onehot_y'], 'test_feature_x': test_set_x, 'test_int_y': group_value['test_int_y'],
            'test_onehot_y': group_value['test_onehot_y']}
        return data_set

    #  train the model
    def trainModel(self, shuffled_groups, select_channels='emg_all'):
        models = []
        results = []
        for group_number, group_value in shuffled_groups.items():
            # extract the dataset
            input_size = group_value['train_feature_x'].shape[2]
            class_number = len(set(group_value['train_int_y']))
            data_set = self.selectSamples(group_value, select_channels)

            # training parameters
            self.model = Raw_Cnn_2d(input_size, class_number).to(self.device)  # move the model to GPU
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # the SGD Optimizers specified in the torch.optim package
            self.loss_fn = torch.nn.CrossEntropyLoss()  # Loss functions expect data in batches

            # dataset of a group
            train_data = EmgDataSet(data_set, 'train')
            self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0)
            test_data = EmgDataSet(data_set, 'test')
            self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0)

            # train and test the dataset for one group
            self.trainOneFold(group_number)
            test_true_labels, test_predict_softmax, test_predict_labels = self.predictTestResults()  # classify the test dataset
            results.append({"true_value": test_true_labels, "predict_softmax": test_predict_softmax, "predict_value": test_predict_labels})
            models.append(self.model.to("cpu"))
        return models, results


# 把tensorboard的内容加上。