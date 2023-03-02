'''
This is a complete code including the use of Tensorboard. The training and test results are reported every 10 epoches,
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
class reduced_Cnn_1d(nn.Module):
    def __init__(self, input_size, class_number):
        super(reduced_Cnn_1d, self).__init__()

        # define layer parameter
        self.conv1_parameter = [32, 3]
        self.conv2_parameter = [32, 3]
        self.conv3_parameter = [32, 3]
        self.linear1_parameter = 500
        self.linear2_parameter = 100

        # define convolutional layer
        self.convolutional_layer = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=self.conv1_parameter[0], kernel_size=self.conv1_parameter[1], dilation=2, stride=2),
            nn.BatchNorm1d(self.conv1_parameter[0]),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=1, padding=0),

            nn.Conv1d(in_channels=self.conv1_parameter[0], out_channels=self.conv2_parameter[0], kernel_size=self.conv2_parameter[1], dilation=2, stride=2),
            nn.BatchNorm1d(self.conv2_parameter[0]),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=1, padding=0),

            nn.Conv1d(in_channels=self.conv2_parameter[0], out_channels=self.conv3_parameter[0], kernel_size=self.conv3_parameter[1], dilation=2, stride=2),
            nn.BatchNorm1d(self.conv3_parameter[0]),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=1, padding=0)
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
        return x


## model summary
# model = reduced_Cnn_1d(2, 13).to('cpu')  # move the model to GPU
# summary(model, input_size=(1024, 2, 350))


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
        self.result_dir = f'D:\Project\pythonProject\Channel_Number\Results\\runs_{timestamp}'

    #  train the model
    def trainModel(self, shuffled_groups, decay_epochs, select_channels='emg_all'):
        models = []
        results = []
        # train and test the dataset for each group
        for group_number, group_value in shuffled_groups.items():
        # for group_number, group_value in {'group_1': shuffled_groups['group_1'], 'group_3': shuffled_groups['group_3']}.items():
            # initialize the tensorboard writer
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(os.path.join(self.result_dir, f'experiment_{timestamp}'))

            # extract the dataset
            input_size = group_value['train_feature_x'].shape[1]
            class_number = len(set(group_value['train_int_y']))
            data_set = self.selectSamples(group_value, select_channels)

            # dataset of a group
            train_data = EmgDataSet(data_set, 'train')
            self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0)
            test_data = EmgDataSet(data_set, 'test')
            self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0)

            # training parameters
            self.model = reduced_Cnn_1d(input_size, class_number).to(self.device)  # move the model to GPU
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.0001)  # initial learning rate and regularization
            self.loss_fn = torch.nn.CrossEntropyLoss()  # Loss functions expect data in batches
            decay_steps = decay_epochs * len(self.train_loader)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_steps, gamma=0.1)  # adjusted learning rate

            # train and test the model of a group
            for epoch_number in range(self.num_epochs):  # loop over each epoch
                self.trainOneEpoch(group_number, epoch_number)
                # classify the test set every 10 epochs
                if (epoch_number + 1) % self.report_period == 0:
                    test_true_labels, test_predict_softmax, test_predict_labels = self.predictTestResults(group_number, epoch_number)
            # log the model structure
            self.writer.add_graph(self.model, next(iter(self.train_loader))[0].to(self.device))

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
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
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
            self.writer.add_scalars('Accuracy', {f'{group_number}_train accuracy': training_accuracy}, epoch_number)
            self.writer.add_scalars('Loss', {f'{group_number}_train loss': train_loss}, epoch_number )
            self.writer.flush()
        else:
            print(f"group: {int(group_number[-1])}, epoch: {epoch_number + 1}, batch: {len(self.train_loader) * (epoch_number + 1)}, "
                f"learning rate: {self.lr_scheduler.get_last_lr()}")

    #  classify test set
    def predictTestResults(self, group_number, epoch_number=0):
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
                test_inputs, test_labels = test_data[0].to(self.device), test_data[1].to(self.device)
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
            self.writer.add_scalars('Accuracy', {f'{group_number}_test accuracy': test_accuracy}, epoch_number)
            self.writer.add_scalars('Loss', {f'{group_number}_test loss': test_loss}, epoch_number)
            self.writer.flush()

        return np.array(test_true_labels), np.array(test_predict_softmax), np.array(test_predict_labels)

    # select specific channels for model training and testing
    def selectSamples(self, group_value, select_channels, bipolar_position=(0, 0)):
        # training dataset
        if select_channels == 'emg_all':
            return group_value
        elif select_channels == 'emg_1':
            train_set_x = group_value['train_feature_x'][:, 0: 65, :, :]
            test_set_x = group_value['test_feature_x'][:, 0: 65, :, :]
        elif select_channels == 'emg_2':
            train_set_x = group_value['train_feature_x'][:, 65: 130, :, :]
            test_set_x = group_value['test_feature_x'][:, 65: 130, :, :]
        elif select_channels == 'bipolar':
            pass
        else:
            raise Exception("No Such Channels")
        data_set = {'train_feature_x': train_set_x, 'train_int_y': group_value['train_int_y'],
            'train_onehot_y': group_value['train_onehot_y'], 'test_feature_x': test_set_x, 'test_int_y': group_value['test_int_y'],
            'test_onehot_y': group_value['test_onehot_y']}
        return data_set


##  loading dataset
class EmgDataSet(Dataset):
    def __init__(self, shuffled_data, mode):
        self.n_samples = shuffled_data[f'{mode}_int_y'].shape[0]
        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(np.transpose(shuffled_data[f'{mode}_feature_x'], (3, 2, 1, 0)))  # size [n_samples, n_channel, features, sequence length]
        self.y_data = torch.from_numpy(shuffled_data[f'{mode}_int_y'])  # size [n_samples]

    def __getitem__(self, index):   # support indexing such that dataset[i] can be used to get i-th sample
        return np.squeeze(self.x_data[index, :, :, :], axis=1), self.y_data[index]  # remove the channel dimension

    def __len__(self):  # we can call len(dataset) to return the size
        return self.n_samples


##  visualize the images
import matplotlib.pyplot as plt
# train_data = EmgDataSet(shuffled_groups['group_0'], 'train')
#
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(train_data), size=(1,)).item()
#     img, label = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis("off")
#     plt.imshow(img.squeeze())
# plt.show()


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)
        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)