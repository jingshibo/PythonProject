##  import
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm

## training
class ModelTesting():
    def __init__(self, model, batch_size):
        #  initialize member variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.model = model.to(self.device)
        self.test_loader = None

    #  test the model
    def testModel(self, test_data):
        dataset = EmgDataSet(test_data)
        self.test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0)
        test_results = []

        self.model.train(False)
        with torch.no_grad():  # close autograd
            # loop over each batch
            for data in tqdm(self.test_loader):
                data = data.to(self.device)
                results = self.model(data).cpu().numpy()
                test_results.append(results)

        return np.vstack(test_results)


##  loading dataset
class EmgDataSet(Dataset):
    def __init__(self, test_emg):
        # here the first column is the class label, the rest are the features
        self.test_data = torch.from_numpy(test_emg)  # size [n_samples, n_channel, length, width]

    def __getitem__(self, index):   # support indexing such that dataset[i] can be used to get i-th sample
        return self.test_data[index, :, :, :]

    def __len__(self):  # we can call len(dataset) to return the size
        return len(self.test_data)


