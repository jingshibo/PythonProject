##  import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from torchinfo import summary

##  model
class CNN_2D(nn.Module):
    def __init__(self, input_size, class_number):
        super(CNN_2D, self).__init__()

        # define layer parameter
        self.linear1_parameter = 600
        self.linear2_parameter = 600
        self.linear3_parameter = 600

        # define dense layer
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=self.linear1_parameter), nn.BatchNorm1d(self.linear1_parameter), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(in_features=self.linear1_parameter, out_features=self.linear2_parameter), nn.BatchNorm1d(self.linear2_parameter), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(in_features=self.linear2_parameter, out_features=self.linear3_parameter), nn.BatchNorm1d(self.linear3_parameter), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(in_features=self.linear3_parameter, out_features=class_number)
        )

    def forward(self, x):
        x = self.linear_layer(x)
        # if self.training is False:
        #     x = F.softmax(x, dim=1)
        return x

##  check model
input_size = 1040
class_number = 13
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN_2D(input_size, class_number).to(device)  # move the model to GPU
summary(model, input_size=(1024, 1040))

##  dataset
class EmgDataSet(Dataset):
    def __init__(self, shuffled_data, mode):
        self.n_samples = shuffled_data[f'{mode}_int_y'].shape[0]
        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(shuffled_data[f'{mode}_feature_x'].astype(np.float32))  # size [n_samples, n_channel, length, width]
        self.y_data = torch.from_numpy(shuffled_data[f'{mode}_int_y'])  # size [n_samples]

    def __getitem__(self, index):   # support indexing such that dataset[i] can be used to get i-th sample
        return self.x_data[index, :], self.y_data[index]

    def __len__(self):  # we can call len(dataset) to return the size
        return self.n_samples


##  training process
# training the model for one epoch
def trainOneFold(training_loader, epochs, group_number, model, loss_fn, optimizer):
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
        print(f"group_number: {group_number}, epoch: {epoch_number}, batch: {i}, loss: {loss:>7f}")

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

    return np.array(test_true_labels), np.array(test_predict_softmax), np.array(test_predict_labels)

def trainModel(shuffled_groups, epochs):
    models = []
    results = []
    for group_number, group_value in shuffled_groups.items():
        model = CNN_2D(input_size, class_number).to(device)  # move the model to GPU
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-08, weight_decay=0.00001)
        loss_fn = torch.nn.CrossEntropyLoss()

        train_data = EmgDataSet(group_value, 'train')
        train_loader = DataLoader(train_data, batch_size=1024, shuffle=True, pin_memory=True, num_workers=0)
        trainOneFold(train_loader, epochs, group_number, model, loss_fn, optimizer)  # train the dataset for one group

        # after finish the training for each group of data
        test_data = EmgDataSet(group_value, 'test')
        test_loader = DataLoader(test_data, batch_size=5000, shuffle=False, pin_memory=True, num_workers=0)
        test_true_labels, test_predict_softmax, test_predict_labels = predictTestResults(test_loader, model)  # classify the test dataset

        results.append({"true_value": test_true_labels, "predict_softmax": test_predict_softmax, "predict_value": test_predict_labels})
        models.append(model.to("cpu"))

    return models, results


##
# the SGD Optimizers specified in the torch.optim package
# Loss functions expect data in batches, so we're creating batches of 4
epochs = 100
models, results = trainModel(shuffled_groups, epochs)
model_results = results
