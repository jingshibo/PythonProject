##
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import datetime
import numpy as np
from cGAN_Transformer.Models import Transformer_GAN_Model


## training process
class GanTraining():
    def __init__(self, num_epochs, batch_size, repetition, gen_update_interval, disc_update_interval, decay_epochs, noise_dim, blending_factor_dim):
        #  initialize member variables
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = f'/Conditional_GAN/Others\\runs_{timestamp}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.repetition = repetition
        self.gen_update_interval = gen_update_interval
        self.disc_update_interval = disc_update_interval
        self.decay_epochs = decay_epochs
        self.current_step = 0
        self.gen_losses = []
        self.disc_losses = []
        self.display_step = None
        self.img_channel = None
        self.img_height = None
        self.img_width = None
        self.n_classes = None
        self.epsilon = None  # soft label parameter
        self.gen = None
        self.disc = None
        self.gen_opt = None
        self.disc_opt = None
        self.lr_gen_opt = None
        self.train_loader = None
        self.test_loader = None
        self.loss_fn = None
        self.recon_loss = None
        self.writer = None

    def trainModel(self, train_gan_data, condition_encoding, checkpoint_model_path, checkpoint_result_path, training_parameters, transition_type):
        # input data
        dataset = EMGFusionDataset(train_gan_data, condition_encoding, self.batch_size, self.repetition)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # training model
        self.gen = Transformer_GAN_Model.EMGFusionGenerator(num_conditions=len(condition_encoding)).to(self.device)
        self.disc = Transformer_GAN_Model.EMGFusionDiscriminator(num_conditions=len(condition_encoding)).to(self.device)

        # optimizer
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=1e-4)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=1e-4)

        # loss function
        self.loss_fn = nn.BCELoss()
        self.recon_loss = nn.L1Loss()

        # train the model
        for epoch_number in range(self.num_epochs):  # loop over each epoch
            # train the model
            self.trainOneEpoch(epoch_number)


    def trainOneEpoch(self, epoch_number):
        self.gen.train(True)
        self.disc.train(True)

        for A, B, C, cond in tqdm(self.train_loader):
            A, B, C, cond = A.to(self.device), B.to(self.device), C.to(self.device), cond.to(self.device)

            # Train Discriminator
            fake_C = self.gen(A, B, cond).detach()
            real_label = torch.ones(A.size(0), 1, device=self.device)
            fake_label = torch.zeros(A.size(0), 1, device=self.device)

            disc_real = self.disc(A, B, C, cond)
            disc_fake = self.disc(A, B, fake_C, cond)

            loss_real = self.loss_fn(disc_real, real_label)
            loss_fake = self.loss_fn(disc_fake, fake_label)
            disc_loss = (loss_real + loss_fake) / 2

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()

            # Train Generator
            fake_C = self.gen(A, B, cond)
            pred_fake = self.disc(A, B, fake_C, cond)

            adv_loss = self.loss_fn(pred_fake, real_label)
            l1 = self.recon_loss(fake_C, C)
            gen_loss = adv_loss + 10 * l1  # Weighted sum

            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()

        print(f"Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")


# ------------------------------
# Dataset class to load EMG fusion data
# ------------------------------
class EMGFusionDataset(Dataset):
    def __init__(self, train_gan_data, condition_encoding, batch_size, repetition):
        self.data = []
        for transition_type, group in train_gan_data.items():
            label = condition_encoding[transition_type]  # get the encoded integer value of the condition
            self.data.append({
                'gen_data_1': group['gen_data_1'],
                'gen_data_2': group['gen_data_2'],
                'disc_data': group['disc_data'],
                'condition': label
            })
        self.batch_size = batch_size
        self.repetition = repetition

    def __len__(self):  # number of samples to train per epoch
        return self.batch_size * self.repetition

    def __getitem__(self, idx):
        # Randomly picks 'one group' (condition) from `self.data`
        group = np.random.choice(self.data)

        # Randomly selects one sample in the group from each of:
        A = group['gen_data_1'][np.random.randint(len(group['gen_data_1']))].T  # Transposes each from `[T, 65]` to `[65, T]`
        B = group['gen_data_2'][np.random.randint(len(group['gen_data_2']))].T
        C = group['disc_data'][np.random.randint(len(group['disc_data']))].T

        A = torch.tensor(A, dtype=torch.float32).unsqueeze(0)  # Converts to PyTorch tensors with shape `[1, 65, T]`
        B = torch.tensor(B, dtype=torch.float32).unsqueeze(0)
        C = torch.tensor(C, dtype=torch.float32).unsqueeze(0)
        condition = torch.tensor(group['condition'], dtype=torch.long)
        return A, B, C, condition