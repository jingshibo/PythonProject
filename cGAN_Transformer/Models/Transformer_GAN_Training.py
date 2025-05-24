##
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import os
import datetime
import numpy as np
from cGAN_Transformer.Functions import Storage
from cGAN_Transformer.Models import Transformer_GAN_Model
from torch.utils.tensorboard import SummaryWriter


##
class GanTraining():
    def __init__(self, num_epochs, batch_size, repetition, num_conditions):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = f'D:\Project\pythonProject\cGAN_Transformer\Results\\runs_{timestamp}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = None

        self.num_conditions_for_models = num_conditions
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.repetition = repetition

        self.gp_lambda = None
        self.critic_iterations = None
        self.report_interval_steps = 10
        self.gen_scaler = GradScaler(enabled=(self.device.type == 'cuda'))
        self.critic_scaler = GradScaler(enabled=(self.device.type == 'cuda'))

        self.gen = None
        self.critic = None  # Renamed from disc
        self.gen_opt = None
        self.critic_opt = None
        self.lr_gen_opt = None
        self.lr_disc_opt = None
        self.train_loader = None

    def trainModel(self, train_gan_data, condition_encoding, training_parameters, storage_parameters):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(self.result_dir, f'experiment_{timestamp}'))
        self.num_conditions_for_models = len(condition_encoding)
        dataset = EMGFusionDataset(train_gan_data, condition_encoding, self.batch_size, self.repetition)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=0)
        self.gen = Transformer_GAN_Model.EMGFusionGenerator(num_conditions=self.num_conditions_for_models).to(self.device)
        self.critic = Transformer_GAN_Model.EMGFusionPatchDiscriminator(num_conditions=self.num_conditions_for_models).to(self.device)

        # training parameters
        gen_lr = 0.0003
        disc_lr = 0.0002
        gen_lr_decay_rate = 0.8
        disc_lr_decay_rate = 0.8
        decay_epochs = [50, 75]
        self.critic_iterations = 3  # Number of critic updates per generator update
        self.gp_lambda = 10.0  # GP weight

        # For WGAN, Adam with these betas is common, or RMSprop
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=gen_lr, weight_decay=0, betas=(0.7, 0.999))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=disc_lr, weight_decay=0.0001, betas=(0.7, 0.999))
        self.lr_gen_opt = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.gen_opt, milestones=decay_epochs, gamma=gen_lr_decay_rate)
        self.lr_disc_opt = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.critic_opt, milestones=decay_epochs, gamma=disc_lr_decay_rate)
        models = {'gen': self.gen, 'disc': self.critic}

        for epoch_number in range(self.num_epochs):
            self.lr_disc_opt.step()  # update the learning rate
            self.lr_gen_opt.step()  # update the learning rate
            avg_gen_loss, avg_critic_loss = self.trainOneEpoch(epoch_number)
            print(
                f"Epoch [{epoch_number + 1}/{self.num_epochs}] Avg Gen Loss: {avg_gen_loss:.4f}, Avg Critic Loss: {avg_critic_loss:.4f}, "
                f"gen_lr: {self.lr_gen_opt.get_last_lr()}, disc_lr: {self.lr_disc_opt.get_last_lr()}")
            # set the checkpoints to save models
            if (epoch_number + 1) % 2 == 0:
                print(f"Saved checkpoint at epoch {epoch_number + 1}")
                Storage.saveCheckPointModels(models, storage_parameters, epoch_number + 1)

        # save the final model
        Storage.saveGanModels(models, storage_parameters)
        return self.gen

    def trainOneEpoch(self, epoch_number):
        self.gen.train()
        self.critic.train()

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch_number + 1}/{self.num_epochs}", leave=True)
        total_gen_loss_epoch = 0.0
        total_critic_loss_epoch = 0.0
        num_gen_updates = 0  # Track number of G updates

        for batch_idx, (A, B, real_C, cond_label) in enumerate(loop):
            A = A.to(self.device)
            B = B.to(self.device)
            real_C = real_C.to(self.device);
            cond_label = cond_label.to(self.device)
            batch_number = epoch_number * self.repetition + batch_idx

            self.critic_opt.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                fake_C = self.gen(A, B, cond_label).detach()

                critic_real_scores = self.critic(real_C, A, B, cond_label)
                critic_fake_scores = self.critic(fake_C, A, B, cond_label)

                # WGAN loss for critic: E[D(fake)] - E[D(real)]
                # We want to MAXIMIZE D(real) - D(fake), so we MINIMIZE D(fake) - D(real)
                loss_critic_adv = torch.mean(critic_fake_scores) - torch.mean(critic_real_scores)

                gp = compute_gradient_penalty(self.critic, real_C, fake_C, A, B, cond_label, self.device)
                critic_loss = loss_critic_adv + self.gp_lambda * gp

            self.critic_scaler.scale(critic_loss).backward()
            self.critic_scaler.step(self.critic_opt)
            self.critic_scaler.update()

            total_critic_loss_epoch += critic_loss.item()

            # --- Train Generator ---
            # Update generator less frequently (e.g., every self.critic_iterations steps)
            if batch_idx % self.critic_iterations == 0:
                self.gen_opt.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    fake_C_for_G = self.gen(A, B, cond_label)
                    gen_fake_scores = self.critic(fake_C_for_G, A, B, cond_label)

                    # WGAN loss for generator: -E[D(fake_C)]
                    # We want to MAXIMIZE D(fake_C), so we MINIMIZE -D(fake_C)
                    loss_gen_adv = -torch.mean(gen_fake_scores)
                    gen_loss = loss_gen_adv

                self.gen_scaler.scale(gen_loss).backward()
                self.gen_scaler.step(self.gen_opt)
                self.gen_scaler.update()

                total_gen_loss_epoch += gen_loss.item()
                num_gen_updates += 1

            # Log the average training loss per 10 batches
            if batch_idx % (self.critic_iterations * 3) == 0:
                print(f"Batch [{batch_number}] Gen Loss: {gen_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
                self.writer.add_scalars('Loss', {'Generator Loss': gen_loss.item()}, batch_number)
                self.writer.add_scalars('Loss', {'Discriminator Loss': critic_loss.item()}, batch_number)
                self.writer.flush()

        avg_gen_loss = total_gen_loss_epoch / num_gen_updates if num_gen_updates > 0 else 0
        avg_critic_loss = total_critic_loss_epoch / len(self.train_loader) if len(self.train_loader) > 0 else 0
        return avg_gen_loss, avg_critic_loss


## Calculates the gradient penalty loss for WGAN GP
def compute_gradient_penalty(critic, real_samples, fake_samples, A_ref, B_ref, condition_label, device):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=device)
    # Get random interpolation between real and fake samples
    interpolates_C = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # For conditional GANs, you might also want to interpolate conditioning factors if they vary per sample,
    d_interpolates = critic(interpolates_C, A_ref, B_ref, condition_label)

    # Create a tensor of ones with the same shape as d_interpolates
    fake = torch.ones_like(d_interpolates, requires_grad=False, device=device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates_C, grad_outputs=fake, create_graph=True, retain_graph=True,
        only_inputs=True, )[0]

    gradients = gradients.view(gradients.size(0), -1)  # Flatten gradients
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


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
        C_real = group['disc_data'][np.random.randint(len(group['disc_data']))].T

        A = torch.tensor(A, dtype=torch.float32).unsqueeze(0)  # Converts to PyTorch tensors with shape `[1, 65, T]`
        B = torch.tensor(B, dtype=torch.float32).unsqueeze(0)
        C_real = torch.tensor(C_real, dtype=torch.float32).unsqueeze(0)
        condition = torch.tensor(group['condition'], dtype=torch.long)
        return A, B, C_real, condition