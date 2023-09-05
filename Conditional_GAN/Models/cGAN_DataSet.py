import random
import torch
from torch.utils.data import Dataset


##  loading dataset
class RandomPureEmgDataSet(Dataset):
    """
       Custom Dataset class for GAN training. Each poach has different samples as new data are resampled randomly for every epoch.
       'Pure' means all data in a single batch come from the same time point.
       Notes:
       - Each key in the datasets corresponds to a unique 4D tensor.
       - The class facilitates sampling data from the same key for both generator and discriminator.
       - The `condition` (normalized key value) is also returned for conditional GAN training.
    """

    def __init__(self, train_data, batch_size, sampling_repetition):
        self.batch_size = batch_size
        self.repetition = sampling_repetition  # this refers to the number of batches to repeat for the same time points
        self.keys = list(train_data['gen_data_1'].keys())  # Assuming all three dicts have the same keys
        self.n_class = len(self.keys)

        # size [n_samples, n_channel, length, width]
        self.gen_data_1 = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['gen_data_1'].items()}
        self.gen_data_2 = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['gen_data_2'].items()}
        self.disc_data = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['disc_data'].items()}

        # Precompute the sequence of keys for an epoch.
        random.shuffle(self.keys)  # randomize the order of the keys themselves.
        self.epoch_keys = self.keys * self.repetition  # preserve the order of keys in 4 batches of combination sampling

    def __len__(self):
        # the number of samples obtained in an epoch for each key = self.repetition * self.batch_size
        return self.repetition * self.batch_size * len(self.keys)

    def extract_and_normalize(self, key):  # convert keys into numbers
        max_number = max(int(key.split('_')[-1]) for key in self.keys)
        interval = max_number // (self.n_class - 1)  # timepoint_interval
        # Extract the number from the key string
        number = int(key.split('_')[-1])
        # Normalize the extracted number (to 0, 1, 2, ... 16)
        normalized_value = number // interval
        return normalized_value

    def __getitem__(self, idx):
        # Determine the key for this idx
        key = self.epoch_keys[idx // self.batch_size]  # This ensures that all indices within the same batch use the same key.
        condition = self.extract_and_normalize(key)  # extract the time point integer from the key string

        # Sample random data points from the chosen key
        gen_sample_1, gen_sample_2, disc_sample = self.paired_sample_from_timepoint(key)

        return (condition, gen_sample_1, gen_sample_2), disc_sample

    def paired_sample_from_timepoint(self, timepoint):
        """
            Samples data from gen_data_1_dict, gen_data_2_dict, and disc_data_dict using the provided timepoint,
            ensuring the sample index remains the same within the tensors.
        """
        # Extract the data associated with the provided timepoint
        gen_tensor_1_list = self.gen_data_1[timepoint]
        gen_tensor_2_list = self.gen_data_2[timepoint]
        disc_tensor_list = self.disc_data[timepoint]

        # Randomly select a tensor from gen_data_1
        tensor_idx_1 = random.randint(0, len(gen_tensor_1_list) - 1)
        tensor_1 = gen_tensor_1_list[tensor_idx_1]
        # Sample a data slice from the selected tensor
        sample_idx = random.randint(0, tensor_1.shape[0] - 1)  # Randomly pick a sample index
        gen_sample_1 = tensor_1[sample_idx]

        # For gen_data_2, we can pick any tensor but must use the same sample index
        tensor_idx_2 = random.randint(0, len(gen_tensor_2_list) - 1)
        tensor_2 = gen_tensor_2_list[tensor_idx_2]
        gen_sample_2 = tensor_2[sample_idx]

        # For disc_data, we can pick any tensor but must use the same sample index
        tensor_idx_3 = random.randint(0, len(disc_tensor_list) - 1)
        tensor_3 = disc_tensor_list[tensor_idx_3]
        disc_sample = tensor_3[sample_idx]

        return gen_sample_1, gen_sample_2, disc_sample



class FixedPureEmgDataSet(Dataset):
    '''
        Custom Dataset class for GAN training. Each poach share the same samples because all samples are precomputed during
        initialization and stored, so that the same samples are used in every epoch.
        'Pure' means all data in a single batch come from the same time point.
    '''
    def __init__(self, train_data, batch_size, sampling_repetition):
        self.batch_size = batch_size
        self.repetition = sampling_repetition  # this refers to the number of batches to repeat for the same time points
        self.keys = list(train_data['gen_data_1'].keys())
        self.n_class = len(self.keys)

        # Convert data to Torch tensors
        self.gen_data_1 = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['gen_data_1'].items()}
        self.gen_data_2 = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['gen_data_2'].items()}
        self.disc_data = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['disc_data'].items()}

        # Precompute the sequence of keys for an epoch
        random.shuffle(self.keys)
        self.epoch_keys = self.keys * self.repetition

        # Precompute and store all samples for each key value
        self.precomputed_samples = []
        for time_point in self.epoch_keys:
            condition = self.extract_and_normalize(time_point)
            for _ in range(batch_size):  # generate batch_size samples of the same time_point for a single batch
                gen_sample_1, gen_sample_2, disc_sample = self.paired_sample_from_timepoint(time_point)
                self.precomputed_samples.append(((condition, gen_sample_1, gen_sample_2), disc_sample))

    def __len__(self):
        return len(self.precomputed_samples)

    def __getitem__(self, idx):
        return self.precomputed_samples[idx]

    def extract_and_normalize(self, key):  # convert keys into numbers
        max_number = max(int(key.split('_')[-1]) for key in self.keys)
        interval = max_number // (self.n_class - 1)  # timepoint_interval
        # Extract the number from the key string
        number = int(key.split('_')[-1])
        # Normalize the extracted number (to 0, 1, 2, ... 16)
        normalized_value = number // interval
        return normalized_value

    def paired_sample_from_timepoint(self, time_point):
        """
            Samples data from gen_data_1_dict, gen_data_2_dict, and disc_data_dict using the provided timepoint,
            ensuring the sample index remains the same within the tensors.
        """
        # Extract the data associated with the provided timepoint
        gen_tensor_1_list = self.gen_data_1[time_point]
        gen_tensor_2_list = self.gen_data_2[time_point]
        disc_tensor_list = self.disc_data[time_point]

        # Randomly select a tensor from gen_data_1
        tensor_idx_1 = random.randint(0, len(gen_tensor_1_list) - 1)
        tensor_1 = gen_tensor_1_list[tensor_idx_1]
        # Sample a data slice from the selected tensor
        sample_idx = random.randint(0, tensor_1.shape[0] - 1)  # Randomly pick a sample index
        gen_sample_1 = tensor_1[sample_idx]

        # For gen_data_2, we can pick any tensor but must use the same sample index
        tensor_idx_2 = random.randint(0, len(gen_tensor_2_list) - 1)
        tensor_2 = gen_tensor_2_list[tensor_idx_2]
        gen_sample_2 = tensor_2[sample_idx]

        # For disc_data, we can pick any tensor but must use the same sample index
        tensor_idx_3 = random.randint(0, len(disc_tensor_list) - 1)
        tensor_3 = disc_tensor_list[tensor_idx_3]
        disc_sample = tensor_3[sample_idx]

        return gen_sample_1, gen_sample_2, disc_sample



class FixedMixEmgDataSet(Dataset):
    '''
        Custom Dataset class for GAN training. Each poach share the same samples because all samples are precomputed during
        initialization and stored, so that the same samples are used in every epoch.
        'Mix' means all data in a single batch come from different time point.
    '''
    def __init__(self, train_data, batch_size, sampling_repetition):
        self.batch_size = batch_size
        self.repetition = sampling_repetition  # the number of samples to collect at each time point
        self.keys = list(train_data['gen_data_1'].keys())  # Assuming all three dicts have the same keys
        self.n_class = len(self.keys)
        sorted_keys = sorted(train_data['gen_data_1'].keys(), key=lambda x: int(x.split('_')[-1]))
        interval = int(sorted_keys[1].split('_')[-1]) - int(sorted_keys[0].split('_')[-1])
        self.length = len(train_data['gen_data_1']) * interval

        # Convert data to Torch tensors
        self.gen_data_1 = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['gen_data_1'].items()}
        self.gen_data_2 = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['gen_data_2'].items()}
        self.disc_data = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['disc_data'].items()}

        # Precompute and store all samples
        self.precomputed_samples = []
        total_samples = ((self.length * self.repetition) // self.batch_size + 1) * self.batch_size
        for idx in range(total_samples):
            time_point = self.keys[idx % len(self.keys)]  # make sure each timepoint has the same number of samples
            condition = self.extract_and_normalize(time_point)
            gen_sample_1, gen_sample_2, disc_sample = self.paired_sample_from_timepoint(time_point)
            self.precomputed_samples.append(((condition, gen_sample_1, gen_sample_2), disc_sample))
        # Shuffle precomputed samples once
        random.shuffle(self.precomputed_samples)

    def __len__(self):
        return len(self.precomputed_samples)

    def __getitem__(self, idx):
        return self.precomputed_samples[idx]

    def extract_and_normalize(self, key):  # convert keys into numbers
        max_number = max(int(key.split('_')[-1]) for key in self.keys)
        interval = max_number // (self.n_class - 1)  # timepoint_interval
        # Extract the number from the key string
        number = int(key.split('_')[-1])
        # Normalize the extracted number (to 0, 1, 2, ... 16)
        normalized_value = number // interval
        return normalized_value

    def paired_sample_from_timepoint(self, time_point):
        """
            Samples data from gen_data_1_dict, gen_data_2_dict, and disc_data_dict using the provided timepoint,
            ensuring the sample index remains the same within the tensors.
        """
        # Extract the data associated with the provided timepoint
        gen_tensor_1_list = self.gen_data_1[time_point]
        gen_tensor_2_list = self.gen_data_2[time_point]
        disc_tensor_list = self.disc_data[time_point]

        # Randomly select a tensor from gen_data_1_list
        tensor_idx_1 = random.randint(0, len(gen_tensor_1_list) - 1)
        tensor_1 = gen_tensor_1_list[tensor_idx_1]
        # Sample a data slice from the selected tensor
        sample_idx = random.randint(0, tensor_1.shape[0] - 1)  # Randomly pick a sample index
        gen_sample_1 = tensor_1[sample_idx]

        # For gen_data_2, we can pick any tensor but must use the same sample index
        tensor_idx_2 = random.randint(0, len(gen_tensor_2_list) - 1)
        tensor_2 = gen_tensor_2_list[tensor_idx_2]
        gen_sample_2 = tensor_2[sample_idx]

        # For disc_data, we can pick any tensor but must use the same sample index
        tensor_idx_3 = random.randint(0, len(disc_tensor_list) - 1)
        tensor_3 = disc_tensor_list[tensor_idx_3]
        disc_sample = tensor_3[sample_idx]

        return gen_sample_1, gen_sample_2, disc_sample



class RandomMixEmgDataSet(Dataset):
    """
       Custom Dataset class for GAN training. Each poach has different samples as new data are resampled randomly for every epoch.
       'Mix' means all data in a single batch come from different time point.
       Notes:
       - Each key in the datasets corresponds to a unique 4D tensor.
       - The class facilitates sampling data from the same key for both generator and discriminator.
       - The `condition` (normalized key value) is also returned for conditional GAN training.
    """

    def __init__(self, train_data, batch_size, sampling_repetition):
        self.batch_size = batch_size
        self.repetition = sampling_repetition  # the number of samples to collect at each time point
        self.keys = list(train_data['gen_data_1'].keys())  # Assuming all three dicts have the same keys
        self.n_class = len(self.keys)
        sorted_keys = sorted(train_data['gen_data_1'].keys(), key=lambda x: int(x.split('_')[-1]))
        interval = int(sorted_keys[1].split('_')[-1]) - int(sorted_keys[0].split('_')[-1])
        self.length = len(train_data['gen_data_1']) * interval

        # size [n_samples, n_channel, length, width]
        self.gen_data_1 = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['gen_data_1'].items()}
        self.gen_data_2 = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['gen_data_2'].items()}
        self.disc_data = {key: [torch.from_numpy(arr).float() for arr in value] for key, value in train_data['disc_data'].items()}

        # Adjust the generation of epoch_keys
        temp_keys = self.keys.copy()
        random.shuffle(temp_keys)
        self.epoch_keys = temp_keys

    def __len__(self):
        # the number of samples obtained in an epoch for each key = self.repetition * self.batch_size
        n = (self.length * self.repetition) // self.batch_size + 1
        return n * self.batch_size

    def __getitem__(self, idx):
        # Determine the key for this idx
        key = self.epoch_keys[idx % len(self.epoch_keys)]  # this ensures each timepoint has same number of samples in an epoch.
        condition = self.extract_and_normalize(key)
        # Sample random data points from the chosen key
        gen_sample_1, gen_sample_2, disc_sample = self.paired_sample_from_timepoint(key)

        return (condition, gen_sample_1, gen_sample_2), disc_sample

    def extract_and_normalize(self, key):  # convert keys into numbers
        max_number = max(int(key.split('_')[-1]) for key in self.keys)
        interval = max_number // (self.n_class - 1)  # timepoint_interval
        # Extract the number from the key string
        number = int(key.split('_')[-1])
        # Normalize the extracted number (to 0, 1, 2, ... 16)
        normalized_value = number // interval
        return normalized_value

    def paired_sample_from_timepoint(self, timepoint):
        """
            Samples data from gen_data_1_dict, gen_data_2_dict, and disc_data_dict using the provided timepoint,
            ensuring the sample index remains the same within the tensors.
        """
        # Extract the data associated with the provided timepoint
        gen_tensor_1_list = self.gen_data_1[timepoint]
        gen_tensor_2_list = self.gen_data_2[timepoint]
        disc_tensor_list = self.disc_data[timepoint]

        # Randomly select a tensor from gen_data_1
        tensor_idx_1 = random.randint(0, len(gen_tensor_1_list) - 1)
        tensor_1 = gen_tensor_1_list[tensor_idx_1]
        # Sample a data slice from the selected tensor
        sample_idx = random.randint(0, tensor_1.shape[0] - 1)  # Randomly pick a sample index
        gen_sample_1 = tensor_1[sample_idx]

        # For gen_data_2, we can pick any tensor but must use the same sample index
        tensor_idx_2 = random.randint(0, len(gen_tensor_2_list) - 1)
        tensor_2 = gen_tensor_2_list[tensor_idx_2]
        gen_sample_2 = tensor_2[sample_idx]

        # For disc_data, we can pick any tensor but must use the same sample index
        tensor_idx_3 = random.randint(0, len(disc_tensor_list) - 1)
        tensor_3 = disc_tensor_list[tensor_idx_3]
        disc_sample = tensor_3[sample_idx]

        return gen_sample_1, gen_sample_2, disc_sample


