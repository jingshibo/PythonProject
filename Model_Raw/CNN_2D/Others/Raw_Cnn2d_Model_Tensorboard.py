'''
This is a complete code including the use of Tensorboard. The training results are calculated every batch and reported every epoch,
while the test results are calculated and reported every epoch. However, the running speed is the lowest, only half of the normal one.
'''


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
# model = Raw_Cnn_2d(1, 13).to('cpu')  # move the model to GPU
# summary(model, input_size=(1024, 1, 512, 130))


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
        #  initialize member variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.test_loader = None
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = f'D:\Project\pythonProject\Model_Raw\CNN_2D\Results\\runs_{timestamp}'

    #  conduct training of one epoch
    def trainOneEpoch(self, group_number, epoch_number):
        # predict result statistics
        num_sample = 0  # total number
        num_correct = 0  # correct number
        sum_loss = 0  # summation of loss

        self.model.train(True)
        # loop over each batch
        for batch_number, data in enumerate(self.train_loader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # output size: (batch_size, class_number)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # calculate training accurate value and loss summation from all previous batches
            _, predicted = torch.max(outputs.data, 1)  # use outputs.data to remove the grad of outputs variable
            num_correct += (predicted == labels).sum().item()
            num_sample += predicted.size(0)
            sum_loss += loss.item()  # use loss.item() to remove the grad of loss variable

        # calculate average training accuracy and loss for one epoch
        training_accuracy = num_correct / num_sample
        training_loss = sum_loss / batch_number
        print(f"group: {group_number[-1]}, epoch: {epoch_number + 1}, train accuracy: {training_accuracy:>7f}, batch: "
            f"{(batch_number + 1) * (epoch_number + 1)}, loss: {training_loss:>7f}")

        # Log the average training accuracy and loss per epoch
        self.writer.add_scalars('Training results', {'loss': training_loss, 'accuracy': training_accuracy}, epoch_number)
        self.writer.flush()

    #  classify test set of one epoch
    def predictTestResults(self, group_number, epoch_number):
        # predict result statistics
        test_true_labels = []
        test_predict_softmax = []
        test_predict_labels = []

        self.model.train(False)
        with torch.no_grad():  # close autograd
            # prediction result statistics
            num_sample = 0  # total number
            num_correct = 0  # correct number

            # loop over each batch
            for i, test_data in enumerate(self.test_loader):
                test_inputs, test_labels = test_data[0].to(self.device), test_data[1].to(self.device)
                test_outputs = self.model(test_inputs)
                test_softmax = [F.softmax(output, dim=0).cpu().numpy() for output in test_outputs]
                test_loss = self.loss_fn(test_outputs, test_labels)

                # calculate test accurate value summation from all previous batches
                _, predicted = torch.max(test_outputs.data, 1)  # use outputs.data to remove the grad of outputs variable
                num_correct += (predicted == test_labels).sum().item()
                num_sample += predicted.size(0)

                # combine predict results
                test_predict_labels.extend(predicted.cpu().numpy())
                test_predict_softmax.extend(np.array(test_softmax))
                test_true_labels.extend(test_labels.cpu().numpy())

            # calculate average training accuracy and loss for one group
            test_accuracy = num_correct / num_sample
            print(f"group: {group_number[-1]}, epoch: {epoch_number + 1}, test accuracy: {test_accuracy:>7f}, test loss: {test_loss.item():>7f}")

            # Log the average test accuracy per epoch
            self.writer.add_scalars('Test results', {'test accuracy': test_accuracy, 'test loss': test_loss}, epoch_number)
            self.writer.flush()

        return np.array(test_true_labels), np.array(test_predict_softmax), np.array(test_predict_labels)

    #  train the model
    def trainModel(self, shuffled_groups, select_channels='emg_all'):
        models = []
        results = []
        # train and test the dataset for each group
        for group_number, group_value in shuffled_groups.items():
            # initialize the tensorboard writer
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(os.path.join(self.result_dir, f'experiment_{timestamp}'))

            # extract the dataset
            input_size = group_value['train_feature_x'].shape[2]
            class_number = len(set(group_value['train_int_y']))
            data_set = self.selectSamples(group_value, select_channels)

            # dataset of a group
            train_data = EmgDataSet(data_set, 'train')
            self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0)
            test_data = EmgDataSet(data_set, 'test')
            self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0)

            # training parameters
            self.model = Raw_Cnn_2d(input_size, class_number).to(self.device)  # move the model to GPU
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # the SGD Optimizers specified in the torch.optim package
            self.loss_fn = torch.nn.CrossEntropyLoss()  # Loss functions expect data in batches

            # loop over each epoch
            for epoch_number in range(self.num_epochs):
                self.trainOneEpoch(group_number, epoch_number)
                test_true_labels, test_predict_softmax, test_predict_labels = self.predictTestResults(group_number, epoch_number)  # classify the test dataset
            # log the model structure
            self.writer.add_graph(self.model, next(iter(self.train_loader))[0].to(self.device))

            results.append({"true_value": test_true_labels, "predict_softmax": test_predict_softmax, "predict_value": test_predict_labels})
            models.append(self.model.to("cpu"))
        return models, results

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
