'''
    Using trained cGAN model to estimate blending factors
'''

##  import
import torch
import torch.nn.functional as F
import numpy as np

## training
class ModelTesting():
    def __init__(self, model):
        #  initialize member variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gen = model.to(self.device)
        self.keys = None
        self.n_classes = None
        self.noise_dim = None

    #  estimate blending factors
    def estimateBlendingFactors(self, test_data, noise_dim=0):
        self.keys = list(test_data['gen_data_1'].keys())
        self.n_classes = len(self.keys)
        self.noise_dim = noise_dim
        n_iterations = 1000  # Number of times to calculate blending_factor for each time_point for averaging purpose
        epsilon = 0.90  # one-side label smoothing parameter

        blending_factors = {}
        self.gen.train(False)  # Set the generator to evaluation mode
        with torch.no_grad():  # Disable autograd for performance improvement
            for time_point in self.keys:
                number = self.extract_and_normalize(time_point)
                one_hot_labels = F.one_hot(torch.tensor([number] * n_iterations), self.n_classes).to(self.device)
                # one_hot_labels = epsilon * F.one_hot(torch.tensor([number] * n_iterations), self.n_classes).to(self.device)

                if self.noise_dim > 0:
                    # Generate random noise
                    fake_noise = torch.randn(n_iterations, self.noise_dim, device=self.device)
                    # Concatenate noise and one-hot labels
                    noise_and_labels = torch.cat((fake_noise.float(), one_hot_labels.float()), 1)
                else:
                    noise_and_labels = one_hot_labels.float()
                # Generate blending_factor for all iterations at once
                blending_outputs = self.gen(noise_and_labels, one_hot_labels).cpu().numpy()

                # Compute the average blending_factor over n_iterations
                avg_blending_factor = np.mean(blending_outputs, axis=0, keepdims=True)
                # Store the average blending_factor
                blending_factors[time_point] = avg_blending_factor
        return blending_factors

    # convert keys into numbers
    def extract_and_normalize(self, key):  # convert keys into numbers
        max_number = max(int(key.split('_')[-1]) for key in self.keys)
        interval = max_number // (self.n_classes - 1)  # time point interval
        # Extract the number from the key string
        number = int(key.split('_')[-1])
        # Normalize the extracted number (to 0, 1, 2, ... 16)
        normalized_value = number // interval
        return normalized_value

