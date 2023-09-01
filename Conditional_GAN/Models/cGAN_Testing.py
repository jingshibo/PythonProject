'''
    Using trained cGAN model to estimate blending factors
'''

##  import
import torch
import torch.nn.functional as F


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
        n_iterations = 100  # Number of times to calculate blending_factor for each time_point for averaging purpose

        blending_factors = {}
        self.gen.train(False)  # Set the generator to evaluation mode
        with torch.no_grad():  # Disable autograd for performance improvement
            for time_point in self.keys:
                number = self.extract_and_normalize(time_point)
                one_hot_labels = F.one_hot(torch.tensor(number), self.n_classes).unsqueeze(0).to(self.device)

                # Initialize a variable to store the sum of blending_factors over n_iterations
                sum_blending_factors = 0.0
                for _ in range(n_iterations):
                    if self.noise_dim > 0:
                        # Generate random noise
                        fake_noise = torch.randn(1, self.noise_dim, device=self.device)
                        # Concatenate noise and one-hot labels
                        noise_and_labels = torch.cat((fake_noise.float(), one_hot_labels.float()), 1)
                    else:
                        noise_and_labels = one_hot_labels.float()
                    # Generate blending_factor and add it to the sum
                    sum_blending_factors += self.gen(noise_and_labels).cpu().numpy()

                # Compute the average blending_factor over n_iterations
                avg_blending_factor = sum_blending_factors / n_iterations
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

