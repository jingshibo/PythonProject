import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import os
from Generative_Model.Functions import Model_Storage
from Conditional_GAN import cGAN_Model


## training process
class ModelTraining():
    def __init__(self, num_epochs, batch_size, decay_epochs, display_step=200):
        #  initialize member variables
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = f'D:\Project\pythonProject\Conditional_GAN\Results\\runs_{timestamp}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.decay_epochs = decay_epochs
        self.display_step = display_step
        self.current_step = 0
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
        self.noise_dim = 0


    def trainModel(self, train_dataset, checkpoint_folder_path, select_channels='emg_all'):
        # input data
        dataset = EmgDataSet(train_dataset)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0)
        # to be corrected later
        mnist_shape = (1, 28, 28)
        n_classes = 10
        self.noise_dim = 64

        # training model
        generator_input_dim = self.noise_dim + n_classes
        discriminator_img_chan = mnist_shape[0] + n_classes
        self.gen = cGAN_Model.Generator(input_dim=generator_input_dim).to(self.device)
        self.disc = cGAN_Model.Discriminator(img_chan=discriminator_img_chan).to(self.device)

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
        self.lr_gen_opt = torch.optim.lr_scheduler.StepLR(self.gen_opt, step_size=decay_steps, gamma=lr_decay_rate)  # adjusted learning rate
        self.lr_disc_opt = torch.optim.lr_scheduler.StepLR(self.disc_opt, step_size=decay_steps, gamma=lr_decay_rate)  # adjusted learning rate

        # train the model
        models = {'gen': self.gen, 'disc': self.disc}
        for epoch_number in range(self.num_epochs):  # loop over each epoch
            # train the model
            self.trainOneEpoch(epoch_number)
            # set the checkpoints to save models
            if (epoch_number + 1) % 50 == 0 and (epoch_number + 1) >= 200:
                Model_Storage.saveCheckPointModels(checkpoint_folder_path, epoch_number, models)
                print(f"Saved checkpoint at epoch {epoch_number + 1}")

        return models

    def trainOneEpoch(self, epoch_number, save_model=False):
        self.gen.train(True)
        self.disc.train(True)
        gen_mean_loss = 0
        disc_mean_loss = 0

        for real, labels in tqdm(self.train_loader):
            # image_width = image.shape[3]
            batch_size = len(real)
            real = real.to(self.device)
            labels = labels.to(self.device)

            # to be corrected later
            one_hot_labels = F.one_hot(labels.to(self.device), n_classes)
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

            # Generate fake images
            fake_noise = torch.randn(batch_size, self.noise_dim, device=self.device)  # Get noise corresponding to the current batch_size
            noise_and_labels = torch.cat((fake_noise.float(), one_hot_labels.float()), 1)  # Combine the noise vectors and the one-hot labels for the generator
            fake = self.gen(noise_and_labels)  # Generate the conditioned fake images

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


##  loading dataset
class EmgDataSet(Dataset):
    def __init__(self, train_dataset):
        self.n_samples = train_dataset.shape[0]
        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(train_dataset)  # size [n_samples, n_channel, length, width]
        self.y_data = torch.from_numpy(train_dataset)  # size [n_samples, n_channel, length, width]

    def __getitem__(self, index):  # support indexing such that dataset[i] can be used to get i-th sample
        return self.x_data[index, :, :, :], self.y_data[index]

    def __len__(self):  # we can call len(dataset) to return the size
        return self.n_samples

