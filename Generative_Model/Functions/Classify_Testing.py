'''
This is a complete code including the use of Tensorboard. The training and test results are reported every 10 epoches,
'''

##  import
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


## training
class ModelTesting():
    def __init__(self, model, batch_size):
        #  initialize member variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.model = model.to(self.device)
        self.test_loader = None

    #  train the model
    def testModel(self, shuffled_groups, select_channels='emg_all'):
        results = []
        # train and test the dataset for each group
        for group_number, group_value in shuffled_groups.items():
        # for group_number, group_value in {'group_1': shuffled_groups['group_1'], 'group_3': shuffled_groups['group_3']}.items():

            # dataset of a group
            data_set = self.selectSamples(group_value, select_channels)
            test_data = EmgDataSet(data_set, 'test')
            self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0)

            # test the model of a group
            test_true_labels, test_predict_softmax, test_predict_labels = self.predictTestResults(group_number)
            results.append({"true_value": test_true_labels, "predict_softmax": test_predict_softmax, "predict_value": test_predict_labels})
        return results

    #  classify test set
    def predictTestResults(self, group_number):
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
            print(f"group: {int(group_number[-1])}, test accuracy: {test_accuracy:>7f}")
        return np.array(test_true_labels), np.array(test_predict_softmax), np.array(test_predict_labels)

    # select specific channels for model training and testing
    def selectSamples(self, group_value, select_channels='emg_all', bipolar_position=(0, 0)):
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
        self.x_data = torch.from_numpy(np.transpose(shuffled_data[f'{mode}_feature_x'], (3, 2, 0, 1)))  # size [n_samples, n_channel, length, width]
        self.y_data = torch.from_numpy(shuffled_data[f'{mode}_int_y'])  # size [n_samples]

    def __getitem__(self, index):   # support indexing such that dataset[i] can be used to get i-th sample
        return self.x_data[index, :, :, :], self.y_data[index]

    def __len__(self):  # we can call len(dataset) to return the size
        return self.n_samples
