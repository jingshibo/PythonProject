##  import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from torchinfo import summary


##
from Pre_Processing import Preprocessing
from Model_Raw.CNN_2D.Functions import Raw_Cnn2d_Dataset, Raw_Cnn2d_Model
from Models.Utility_Functions import Data_Preparation, MV_Results_ByGroup
from Model_Sliding.ANN.Functions import Sliding_Ann_Results
import datetime

##  read sensor data and filtering

# basic information
subject = 'Shibo'
version = 1  # the data from which experiment version to process
modes = ['up_down', 'down_up']
# up_down_session = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# down_up_session = [10, 11, 12, 13, 19, 24, 25, 26, 27, 28, 20]
up_down_session = [10, 11]
down_up_session = [10, 11]
sessions = [up_down_session, down_up_session]

# read and filter data
split_parameters = Preprocessing.readSplitParameters(subject, version)
emg_filtered_data = Preprocessing.labelFilteredData(subject, modes, sessions, version, split_parameters, start_position=-1024, end_position=1024)
emg_preprocessed = Data_Preparation.removeSomeSamples(emg_filtered_data)
del emg_filtered_data
fold = 5  # 5-fold cross validation
cross_validation_groups = Data_Preparation.crossValidationSet(fold, emg_preprocessed)
del emg_preprocessed


##  reorganize data
now = datetime.datetime.now()
sliding_window_dataset = Raw_Cnn2d_Dataset.seperateEmgData(cross_validation_groups, separation_window_size=512, increment=64)
del cross_validation_groups
normalized_groups = Raw_Cnn2d_Dataset.combineNormalizedDataset(sliding_window_dataset)
del sliding_window_dataset
shuffled_groups = Raw_Cnn2d_Dataset.shuffleTrainingSet(normalized_groups)
del normalized_groups
print(datetime.datetime.now() - now)


##  model
class ConvNet(nn.Module):
    def __init__(self, input_channel, class_number):
        super(ConvNet, self).__init__()

        # define layer parameter
        self.conv1_parameter = [32, 7]
        self.conv2_parameter = [64, 5]
        self.conv3_parameter = [128, 3]
        self.linear1_parameter = 1000
        self.linear2_parameter = 100

        # define convolutional layer
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=self.conv1_parameter[0], kernel_size=self.conv1_parameter[1], stride=1),
            nn.BatchNorm2d(self.conv1_parameter[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=self.conv1_parameter[0], out_channels=self.conv2_parameter[0], kernel_size=self.conv2_parameter[1], stride=1),
            nn.BatchNorm2d(self.conv2_parameter[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=self.conv2_parameter[0], out_channels=self.conv3_parameter[0], kernel_size=self.conv3_parameter[1], stride=1),
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
        # if self.training is False:
        #     x = F.softmax(x, dim=1)
        return x

##
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.LazyLinear(120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 13)
#     def forward(self, x):
#         # -> n, 3, 32, 32
#         x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
#         x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
#         x = x.view(x.size(0), -1)            # -> n, 400
#         x = F.relu(self.fc1(x))               # -> n, 120
#         x = F.relu(self.fc2(x))               # -> n, 84
#         x = self.fc3(x)                       # -> n, 10
#         return x



##  check model
net = ConvNet(1,13)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = net.to(device)  # move the model to GPU
print(net)

##
summary(model, input_size=(512, 1, 512, 130))

##  dataset
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


##  training process
# training the model for one epoch
def trainOneGroup(training_loader, epochs, group_number, model, loss_fn, optimizer):
    model.train(True)
    # repeat training for epochs times
    for epoch_number in range(epochs):
        # loop over for one epoch
        for i, data in enumerate(training_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"group_number: {group_number}, epoch: {epoch_number}, loss: {loss:>7f}")

def predictTestResults(test_loader, model):
    model.train(False)
    test_true_labels = []
    test_predict_softmax = []
    test_predict_labels = []
    with torch.no_grad():  # close autograd
        for i, test_data in enumerate(test_loader):
            test_inputs, test_labels = test_data[0].to(device), test_data[1].to(device)
            predict_softmax = model(test_inputs)
            predict_label = np.argmax(predict_softmax.cpu().numpy(), axis=-1)  # return predicted labels
            test_true_labels.extend(test_labels.cpu().numpy())
            test_predict_softmax.extend(predict_softmax.cpu().numpy())
            test_predict_labels.extend(predict_label)

    return test_true_labels, test_predict_softmax, test_predict_labels

def trainModel(shuffled_groups, epochs, model, loss_fn):
    models = []
    results = []
    for group_number, group_value in shuffled_groups.items():
        train_data = EmgDataSet(group_value, 'train')
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
        trainOneGroup(train_loader, epochs, group_number, model, loss_fn, optimizer)  # train the dataset for one group

        # after finish the training for each group of data
        test_data = EmgDataSet(group_value, 'test')
        test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)
        test_true_labels, test_predict_softmax, test_predict_labels = predictTestResults(test_loader, model)  # classify the test dataset

        results.append({"true_value": test_true_labels, "predict_softmax": test_predict_softmax, "predict_value": test_predict_labels})
        models.append(model.to("cpu"))

    return models, results


##
# the SGD Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Loss functions expect data in batches, so we're creating batches of 4
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 10
models, results = trainModel(shuffled_groups, epochs, model, loss_fn)
