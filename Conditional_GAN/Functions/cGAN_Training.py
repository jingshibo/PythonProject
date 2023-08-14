import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import datetime
from Cycle_GAN.Functions import Model_Storage
from Conditional_GAN.Functions import cGAN_Model
import random


## training process
class ModelTraining():
    def __init__(self, num_epochs, batch_size, sampling_repetition, decay_epochs, noise_dim):
        #  initialize member variables
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = f'D:\Project\pythonProject\Conditional_GAN\Results\\runs_{timestamp}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.sampling_repetition = sampling_repetition
        self.noise_dim = noise_dim
        self.decay_epochs = decay_epochs
        self.current_step = 0
        self.display_step = None
        self.img_channel = None
        self.img_height = None
        self.img_width = None
        self.n_classes = None
        self.gen = None
        self.disc = None
        self.gen_opt = None
        self.disc_opt = None
        self.lr_gen_opt = None
        self.lr_disc_opt = None
        self.loss_fn = None
        self.train_loader = None
        self.test_loader = None
        self.writer = None

    def trainModel(self, train_data, checkpoint_model_path, checkpoint_result_path, select_channels='emg_all'):
        # input data
        dataset = EmgDataSet(train_data, self.batch_size, self.sampling_repetition)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0)
        self.img_channel = train_data['gen_data_1']['timepoint_0'][0].shape[1]
        self.img_height = train_data['gen_data_1']['timepoint_0'][0].shape[2]
        self.img_width = train_data['gen_data_1']['timepoint_0'][0].shape[3]
        self.n_classes = len(list(train_data['gen_data_1'].keys()))
        self.display_step = int(len(dataset) / self.batch_size)

        # training model
        generator_input_dim = self.noise_dim + self.n_classes
        discriminator_input_channel = self.img_channel + self.n_classes
        self.gen = cGAN_Model.Generator(generator_input_dim, self.img_height, self.img_width, img_chan=self.img_channel).to(self.device)
        self.disc = cGAN_Model.Discriminator(discriminator_input_channel).to(self.device)

        # training parameters
        lr = 0.001  # initial learning rate
        lr_decay_rate = 0.95
        weight_decay = 0.0000
        beta = (0.7, 0.999)
        decay_steps = self.decay_epochs * len(self.train_loader)

        # optimizer
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, weight_decay=weight_decay, betas=beta)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr, weight_decay=weight_decay, betas=beta)

        # loss function
        criterion = nn.BCEWithLogitsLoss()
        self.loss_fn = cGAN_Model.LossFunction(criterion)

        # learning rate scheduler
        self.lr_gen_opt = torch.optim.lr_scheduler.StepLR(self.gen_opt, step_size=decay_steps,
            gamma=lr_decay_rate)  # adjusted learning rate
        self.lr_disc_opt = torch.optim.lr_scheduler.StepLR(self.disc_opt, step_size=decay_steps,
            gamma=lr_decay_rate)  # adjusted learning rate

        # train the model
        models = {'gen': self.gen, 'disc': self.disc}
        for epoch_number in range(self.num_epochs):  # loop over each epoch
            # train the model
            self.trainOneEpoch(epoch_number)
            # set the checkpoints to save models
            if (epoch_number + 1) % 50 == 0 and (epoch_number + 1) >= 50:
                Model_Storage.saveCheckPointModels(checkpoint_model_path, epoch_number + 1, models)
                # estimate blending factors
                blending_factors = self.estimateBlendingFactors(dataset, train_data)
                Model_Storage.saveCheckPointCGanResults(checkpoint_result_path, epoch_number + 1, blending_factors)
                print(f"Saved checkpoint at epoch {epoch_number + 1}")

        # estimate blending factors
        blending_factors = self.estimateBlendingFactors(dataset, train_data)

        return models, blending_factors

    def trainOneEpoch(self, epoch_number):
        self.gen.train(True)
        self.disc.train(True)
        gen_mean_loss = 0
        disc_mean_loss = 0

        for gen_data, disc_data in tqdm(self.train_loader):
            # image_width = image.shape [3]
            cur_batch_size = len(disc_data)
            condition = gen_data[0].to(self.device)
            gen_data_1 = gen_data[1].to(self.device)
            gen_data_2 = gen_data[2].to(self.device)
            real = disc_data.to(self.device)

            # convert condition into one hot vectors
            one_hot_labels = F.one_hot(condition, self.n_classes)
            # adding two additional dimensions to the one-hot encoded labels (size [batch_size, n_classes, 1, 1])
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            # match the spatial dimensions of the image.(size [batch_size, n_classes, image_height, image_width])
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, self.img_height, self.img_width)

            # Generate fake images
            if self.noise_dim > 0:
                # Get noise corresponding to the current batch_size
                fake_noise = torch.randn(cur_batch_size, self.noise_dim, device=self.device)
                # Combine the noise vectors and the one-hot labels for the generator
                noise_and_labels = torch.cat((fake_noise.float(), one_hot_labels.float()), 1)
            else:
                noise_and_labels = one_hot_labels.float()
            blending_factors = self.gen(noise_and_labels)  # Estimate blending factors
            # to be corrected as well as the generator model output
            fake = blending_factors * gen_data_1 + (1 - blending_factors) * gen_data_2  # Generate the conditioned fake images

            # Update the discriminator
            self.disc_opt.zero_grad()  # Zero out the discriminator gradients
            disc_loss = self.loss_fn.get_disc_loss(fake, real, image_one_hot_labels, self.disc)
            disc_loss.backward()  # Update gradients
            self.disc_opt.step()  # Update optimizer
            self.lr_disc_opt.step()  # update the learning rate

            # Update the generator
            self.gen_opt.zero_grad()
            gen_loss = self.loss_fn.get_gen_loss(fake, image_one_hot_labels, self.disc)
            gen_loss.backward()
            self.gen_opt.step()
            self.lr_gen_opt.step()  # update the learning rate

            # Keep track of the average discriminator loss
            disc_mean_loss += disc_loss.item() / self.display_step
            # Keep track of the average generator loss
            gen_mean_loss += gen_loss.item() / self.display_step

            if self.current_step % self.display_step == 0 and self.current_step > 0:
                print(f"Epoch {epoch_number}: Step {self.current_step}: Generator loss: {gen_mean_loss}, Discriminator loss: "
                      f"{disc_mean_loss}, learning rate: {self.lr_gen_opt.get_last_lr()}")
                gen_mean_loss = 0
                disc_mean_loss = 0

            self.current_step += 1

    def estimateBlendingFactors(self, dataset, train_data):
        blending_factors = {}
        self.gen.train(False)
        with torch.no_grad():  # close autograd
            for time_point in list(train_data['gen_data_1'].keys()):
                number = dataset.extract_and_normalize(time_point)
                one_hot_labels = F.one_hot(torch.tensor(number), self.n_classes).to(self.device)
                if self.noise_dim > 0:
                    # Get noise corresponding to the current batch_size
                    fake_noise = torch.randn(1, self.noise_dim, device=self.device)
                    # Combine the noise vectors and the one-hot labels for the generator
                    noise_and_labels = torch.cat((fake_noise.float(), one_hot_labels.float()), 1)
                else:
                    noise_and_labels = one_hot_labels.float()
                blending_factors[time_point] = self.gen(noise_and_labels).cpu().numpy()
        return blending_factors


##  loading dataset
class EmgDataSet(Dataset):
    """
       Custom Dataset class for GAN training.
       Notes:
       - Each key in the datasets corresponds to a unique 4D tensor.
       - The class facilitates sampling data from the same key for both generator and discriminator.
       - The `condition` (normalized key value) is also returned for conditional GAN training.
    """

    def __init__(self, train_data, batch_size, sampling_repetition):
        self.batch_size = batch_size
        self.repetition = sampling_repetition  # the number of batches to repeat the combination sampling for the same time points
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
        key = self.epoch_keys[idx // self.batch_size]  # This ensures that all indices within a particular range of 1024 use the same key.
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
