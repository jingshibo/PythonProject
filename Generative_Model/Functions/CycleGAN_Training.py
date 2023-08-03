import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import os
from Generative_Model.Functions import CycleGAN_Model
from Generative_Model.Functions import Model_Storage


## training process
class ModelTraining():
    def __init__(self, num_epochs, batch_size, decay_epochs, display_step=200):
        #  initialize member variables
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = f'D:\Project\pythonProject\Model_Raw\CNN_2D\Results\\runs_{timestamp}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.decay_epochs = decay_epochs
        self.display_step = display_step
        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0
        self.current_step = 0
        self.gen_AB = None
        self.gen_BA = None
        self.disc_A = None
        self.disc_B = None
        self.gen_opt = None
        self.disc_A_opt = None
        self.disc_B_opt = None
        self.loss_fn = None
        self.lr_gen_opt = None
        self.lr_disc_A_opt = None
        self.lr_disc_B_opt = None
        self.train_loader = None
        self.test_loader = None
        self.writer = None

    def trainModel(self, old_data, new_data, checkpoint_folder_path, select_channels='emg_all'):
        models = []
        # initialize the tensorboard writer
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(self.result_dir, f'experiment_{timestamp}'))

        # train dataset
        dataset = EmgDataSet(old_data, new_data)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0)
        dim_A = old_data.shape[1]  # the number of channels of the images in pile A
        dim_B = new_data.shape[1]  # the number of channels of the images in pile B

        # training model
        self.gen_AB = CycleGAN_Model.Generator(dim_A, dim_B).to(self.device)
        self.gen_BA = CycleGAN_Model.Generator(dim_B, dim_A).to(self.device)
        self.disc_A = CycleGAN_Model.Discriminator(dim_A).to(self.device)
        self.disc_B = CycleGAN_Model.Discriminator(dim_B).to(self.device)

        # training parameters
        lr = 0.001  # initial learning rate
        lr_decay_rate = 0.9
        weight_decay = 0.0000
        beta = (0.5, 0.999)
        decay_steps = self.decay_epochs * len(self.train_loader)

        # optimizer
        self.gen_opt = torch.optim.Adam(list(self.gen_AB.parameters()) + list(self.gen_BA.parameters()), lr=lr, weight_decay=weight_decay,
            betas=beta)
        self.disc_A_opt = torch.optim.Adam(self.disc_A.parameters(), lr=lr, weight_decay=weight_decay, betas=beta)
        self.disc_B_opt = torch.optim.Adam(self.disc_B.parameters(), lr=lr, weight_decay=weight_decay, betas=beta)

        # loss function
        adv_criterion = nn.MSELoss()  # an adversarial loss function to keep track of how well the GAN is fooling the discriminator
        recon_criterion = nn.L1Loss()  # a loss function that rewards similar images to the ground truth, which "reconstruct" the image
        self.loss_fn = CycleGAN_Model.LossFunction(adv_criterion, recon_criterion)

        # learning rate scheduler
        self.lr_gen_opt = torch.optim.lr_scheduler.StepLR(self.gen_opt, step_size=decay_steps,
            gamma=lr_decay_rate)  # adjusted learning rate
        self.lr_disc_A_opt = torch.optim.lr_scheduler.StepLR(self.disc_A_opt, step_size=decay_steps,
            gamma=lr_decay_rate)  # adjusted learning rate
        self.lr_disc_B_opt = torch.optim.lr_scheduler.StepLR(self.disc_B_opt, step_size=decay_steps,
            gamma=lr_decay_rate)  # adjusted learning rate

        # train the model
        for epoch_number in range(self.num_epochs):  # loop over each epoch
            # train the model
            self.trainOneEpoch(epoch_number)
            models = {'gen_AB': self.gen_AB.to("cpu"), 'gen_BA': self.gen_BA.to("cpu"), 'disc_A': self.disc_A.to("cpu"),
                'disc_B': self.disc_B.to("cpu")}
            # set the checkpoints to save models
            if (epoch_number + 1) % 50 == 0 and (epoch_number + 1) >= 200:
                Model_Storage.saveCheckPointModels(checkpoint_folder_path, epoch_number, models)
                print(f"Saved checkpoint at {epoch_number + 1}")

        # test the model (still use the train data here to test)
        test_results = []
        self.gen_BA.train(False)
        with torch.no_grad():  # close autograd
            # loop over each batch
            for real_A, real_B in tqdm(self.train_loader):
                real_B = real_B.to(self.device)
                results = self.gen_BA(real_B).cpu().numpy()
                test_results.append(results)
        generated_data = np.vstack(test_results)

        return models, generated_data

    def trainOneEpoch(self, epoch_number, save_model=False):
        self.gen_AB.train(True)
        self.gen_BA.train(True)
        self.disc_A.train(True)
        self.disc_B.train(True)

        for real_A, real_B in tqdm(self.train_loader):
            # image_width = image.shape[3]
            cur_batch_size = len(real_A)
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            # Update discriminator A ###
            self.disc_A_opt.zero_grad()  # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_A = self.gen_BA(real_B)
            disc_A_loss = self.loss_fn.get_disc_loss(real_A, fake_A, self.disc_A)
            disc_A_loss.backward()  # Update gradients
            self.disc_A_opt.step()  # Update optimizer
            self.lr_disc_A_opt.step()  # update the learning rate

            # Update discriminator B ###
            self.disc_B_opt.zero_grad()  # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_B = self.gen_AB(real_A)
            disc_B_loss = self.loss_fn.get_disc_loss(real_B, fake_B, self.disc_B)
            disc_B_loss.backward()  # Update gradients
            self.disc_B_opt.step()  # Update optimizer
            self.lr_disc_B_opt.step()  # update the learning rate

            # Update generator ###
            self.gen_opt.zero_grad()
            gen_loss = self.loss_fn.get_gen_loss(real_A, real_B, self.gen_AB, self.gen_BA, self.disc_A, self.disc_B)
            gen_loss.backward()  # Update gradients
            self.gen_opt.step()  # Update optimizer
            self.lr_gen_opt.step()  # update the learning rate

            # Keep track of the average discriminator loss
            self.mean_discriminator_loss += disc_A_loss.item() / self.display_step
            # Keep track of the average generator loss
            self.mean_generator_loss += gen_loss.item() / self.display_step

            # Visualization code ###
            if self.current_step % self.display_step == 0 and self.current_step != 0:
                print(f"Epoch {epoch_number}: Step {self.current_step}: Generator (U-Net) loss: {self.mean_generator_loss}, Discriminator "
                      f"loss: "f"{self.mean_discriminator_loss}, learning rate: {self.lr_gen_opt.get_last_lr()}")
                self.mean_generator_loss = 0
                self.mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    torch.save(
                        {'gen_AB': self.gen_AB.state_dict(), 'gen_BA': self.gen_BA.state_dict(), 'gen_opt': self.gen_opt.state_dict(),
                            'disc_A': self.disc_A.state_dict(), 'disc_A_opt': self.disc_A_opt.state_dict(),
                            'disc_B': self.disc_B.state_dict(), 'disc_B_opt': self.disc_B_opt.state_dict()},
                        f"cycleGAN_{self.current_step}.pth")
            self.current_step += 1


##  loading dataset
class EmgDataSet(Dataset):
    def __init__(self, old_emg, new_emg):
        # here the first column is the class label, the rest are the features
        self.old_data = torch.from_numpy(old_emg)  # size [n_samples, n_channel, length, width]
        self.new_data = torch.from_numpy(new_emg)  # size [n_sampl    es, n_channel, length, width]

    def __getitem__(self, index):  # support indexing such that dataset[i] can be used to get i-th sample
        return self.old_data[index, :, :, :], self.new_data[index, :, :, :]

    def __len__(self):  # we can call len(dataset) to return the size
        return min(len(self.old_data), len(self.new_data))
