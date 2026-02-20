##
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import os
import gc
import datetime
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import truncnorm
from cGAN_Transformer.Functions import Storage
from cGAN_Transformer.Models import Transformer_GAN_Model
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision.transforms as transforms
from cGAN_Transformer.Functions import Plot_Raw_Data


##
class GanTraining():
    def __init__(self, num_epochs, num_sample_per_condition, num_batch_per_epoch):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = f'D:\Project\pythonProject\cGAN_Transformer\Results\\runs_{timestamp}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = None
        self.num_epochs = num_epochs
        self.num_batch_per_epoch = num_batch_per_epoch
        self.num_sample_per_condition = num_sample_per_condition

        self.gp_lambda = None
        self.lambda_L1 = None
        self.critic_iterations = None
        self.lambda_l1_decay_epochs = None
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
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss_calculator = VGG19PerceptualLoss(feature_layers={'relu3_1': 11}, device='cuda')


    def trainModel(self, train_gan_data, transition_encoding, training_parameters, storage_parameters, gen_model):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(self.result_dir, f'experiment_{timestamp}'))

        num_conditions = len(transition_encoding) * training_parameters['num_window_per_transition']
        batch_size = self.num_sample_per_condition * num_conditions
        dataset = EMGFusionDataset(train_gan_data, transition_encoding, training_parameters['num_window_per_transition'],
            training_parameters['window_length'], training_parameters['window_increment'], training_parameters['window_shift'],
            self.num_sample_per_condition, self.num_batch_per_epoch)
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
        # if gen_model == 'one_factor':
        #     self.gen = Transformer_GAN_Model.EMGFusionOneFactorGenerator(num_conditions).to(self.device)
        # elif gen_model == 'two_factors':
        #     self.gen = Transformer_GAN_Model.EMGFusionTwoFactorGenerator(num_conditions).to(self.device)
        # else:
        #     raise Exception
        self.gen = Transformer_GAN_Model.EMGFusionTwoFactorGenerator(num_conditions).to(self.device)
        self.critic = Transformer_GAN_Model.EMGFusionPatchDiscriminator(num_conditions).to(self.device)

        # training parameters
        gen_lr = 0.0003
        disc_lr = 0.0002
        gen_lr_decay_rate = 0.7
        disc_lr_decay_rate = 0.7
        decay_epochs = [10, 20, 30, 50, 75]
        self.critic_iterations = 3  # Number of critic updates per generator update 5
        self.gp_lambda = 10.0  # GP weight
        self.lambda_L1 = 100  # L1 weight
        self.lambda_l1_decay_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        # For WGAN, Adam with these betas is common, or RMSprop
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=gen_lr, weight_decay=0, betas=(0.5, 0.999))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=disc_lr, weight_decay=0.0001, betas=(0.5, 0.999))
        self.lr_gen_opt = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.gen_opt, milestones=decay_epochs, gamma=gen_lr_decay_rate)
        self.lr_disc_opt = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.critic_opt, milestones=decay_epochs, gamma=disc_lr_decay_rate)
        models = {'gen': self.gen, 'disc': self.critic}

        # log the model structure (use a tiny batch for tracing)
        A, B, C_real, cond = next(iter(self.train_loader))
        A = A.to(self.device)
        B = B.to(self.device)
        C_real = C_real.to(self.device)
        cond = cond.to(self.device)

        # Use a small subset to avoid huge tracing graphs / OOM
        A, B, C_real, cond = A[:2], B[:2], C_real[:2], cond[:2]

        self.gen.eval()
        self.critic.eval()
        with torch.no_grad():
            # Generator: expects (A, B, condition)
            self.writer.add_graph(self.gen, (A, B, cond))

            # Critic on real: expects (C_sample, condition_label)
            self.writer.add_graph(self.critic, (C_real, cond))

            # (Optional) Critic on fake as well
            C_fake, _ = self.gen(A, B, cond)
            self.writer.add_graph(self.critic, (C_fake, cond))

        self.gen.train()
        self.critic.train()

        for epoch_number in range(self.num_epochs):
            avg_gen_loss, avg_critic_loss = self.trainOneEpoch(epoch_number)
            self.lr_disc_opt.step()  # update the learning rate
            self.lr_gen_opt.step()  # update the learning rate
            print(f"Epoch [{epoch_number + 1}/{self.num_epochs}], gen_lr: {self.lr_gen_opt.get_last_lr()}, "
                f"disc_lr: {self.lr_disc_opt.get_last_lr()}, Avg Gen Loss: {avg_gen_loss:.4f}, Avg Critic Loss: {avg_critic_loss:.4f}")
            # set the checkpoints to save models
            if (epoch_number + 1) % 5 == 0:
                print(f"Saved checkpoint at epoch {epoch_number + 1}")
                Storage.saveCheckPointModels(models, storage_parameters, epoch_number + 1, gen_model)

        # save the final model
        Storage.saveGanModels(models, gen_model, storage_parameters)

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)
        return models

    def trainOneEpoch(self, epoch_number):
        self.gen.train()
        self.critic.train()

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch_number + 1}/{self.num_epochs}", leave=True)
        total_gen_loss_epoch = 0.0
        total_critic_loss_epoch = 0.0
        num_gen_updates = 0  # Track number of G updates
        # adjust reconstruction weight
        if (epoch_number + 1) in self.lambda_l1_decay_epochs:
            self.lambda_L1 = self.lambda_L1 // 2

        for batch_idx, (A, B, real_C, cond_label) in enumerate(loop):
            A = A.to(self.device)
            B = B.to(self.device)
            real_C = real_C.to(self.device)
            cond_label = cond_label.to(self.device)
            batch_number = epoch_number * self.num_batch_per_epoch + batch_idx

            self.critic_opt.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                fake_C, _ = self.gen(A, B, cond_label)
                critic_real_scores = self.critic(real_C, cond_label)
                critic_fake_scores = self.critic(fake_C.detach(), cond_label)

                # WGAN loss for critic: E[D(fake)] - E[D(real)], We want to MAXIMIZE D(real) - D(fake), so we MINIMIZE D(fake) - D(real)
                # loss_critic_adv = torch.mean(critic_fake_scores) - torch.mean(critic_real_scores)
                # gp = self.compute_gradient_penalty(self.critic, real_C, fake_C, cond_label, self.device)
                # critic_loss = loss_critic_adv + self.gp_lambda * gp

                # LSGANloss for critic
                real_labels = torch.full_like(critic_real_scores, 0.9, device=self.device)
                loss_disc_real = self.mse_loss(critic_real_scores, real_labels)
                fake_labels = torch.full_like(critic_fake_scores, 0.1, device=self.device)
                loss_disc_fake = self.mse_loss(critic_fake_scores, fake_labels)
                critic_loss = loss_disc_real + loss_disc_fake

            self.critic_scaler.scale(critic_loss).backward()
            self.critic_scaler.step(self.critic_opt)
            self.critic_scaler.update()

            total_critic_loss_epoch += critic_loss.item()

            # --- Train Generator ---
            # Update generator less frequently (e.g., every self.critic_iterations steps)
            if batch_idx % self.critic_iterations == 0:
                self.gen_opt.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    fake_C_for_G, _ = self.gen(A, B, cond_label)
                    gen_fake_scores = self.critic(fake_C_for_G, cond_label)

                    # WGAN loss for generator: -E[D(fake_C)], We want to MAXIMIZE D(fake_C), so we MINIMIZE -D(fake_C)
                    # gen_adv = -torch.mean(gen_fake_scores)
                    # # recon_loss = self.l1_loss(fake_C_for_G, real_C)
                    # recon_loss = self.perceptual_loss_calculator(fake_C_for_G, real_C)
                    # # recon_loss = self.condition_average_reconstruction_loss(fake_C_for_G, real_C, cond_label, loss_fn='VGG19Loss')
                    # gen_loss = gen_adv + self.lambda_L1 * recon_loss

                    # BCEWithLogitsLoss for generator. Generator wants to classify fakes as REAL, so the target labels are real_labels
                    real_labels_for_gen = torch.full_like(gen_fake_scores, 1, device=self.device)
                    gen_adv = self.mse_loss(gen_fake_scores, real_labels_for_gen)
                    # recon_loss = self.mse_loss(fake_C_for_G, real_C)
                    # recon_loss = self.l1_loss(fake_C_for_G, real_C)
                    # recon_loss = self.perceptual_loss_calculator(fake_C_for_G, real_C)
                    recon_loss = self.condition_average_reconstruction_loss(fake_C_for_G, real_C, cond_label, loss_fn='VGG19Loss')
                    gen_loss = gen_adv + self.lambda_L1 * recon_loss

                self.gen_scaler.scale(gen_loss).backward()
                self.gen_scaler.step(self.gen_opt)
                self.gen_scaler.update()

                total_gen_loss_epoch += gen_loss.item()
                num_gen_updates += 1

            # Log the average training loss per 10 batches
            if batch_idx % (self.critic_iterations * 3) == 0:
                print(f"gene_adv: {gen_adv.item():.4f}, recon_loss: {recon_loss.item():.4f}, "
                      f"Critic Loss: {critic_loss.item():.4f}, Gen Loss: {gen_loss.item():.4f}, lambda: {self.lambda_L1}")

                self.writer.add_scalars('Loss', {'Generator Loss': gen_loss.item()}, batch_number)
                self.writer.add_scalars('Loss', {'Discriminator Loss': critic_loss.item()}, batch_number)
                self.writer.add_scalars('Loss', {'gene_adv Loss': gen_adv.item()}, batch_number)
                self.writer.add_scalars('Loss', {'reconstruction Loss': recon_loss.item()}, batch_number)
                self.writer.flush()

        avg_gen_loss = total_gen_loss_epoch / num_gen_updates if num_gen_updates > 0 else 0
        avg_critic_loss = total_critic_loss_epoch / len(self.train_loader) if len(self.train_loader) > 0 else 0
        return avg_gen_loss, avg_critic_loss


    ## calculate average real images per condition in a batch for comparison with fake images
    def condition_average_reconstruction_loss(self, generated, real, conditions, loss_fn='l1'):
        assert generated.shape == real.shape
        B, _, C, T = generated.shape
        device = generated.device
        unique_conditions = torch.unique(conditions)

        # Compute average real image per condition
        avg_real_by_condition = {}
        for cond in unique_conditions:
            mask = (conditions == cond)
            real_cond = real[mask]
            # avg_real = real_cond.mean(dim=0, keepdim=False)  # shape: [1, C, T]
            avg_real = real_cond.max(dim=0, keepdim=False).values
            avg_real_by_condition[int(cond.item())] = avg_real.to(device)

        # Construct batch of condition-averaged real targets
        avg_real_targets = torch.stack([avg_real_by_condition[int(c.item())] for c in conditions])  # shape: [B, 1, C, T]

        # average_value = avg_real_targets.to("cpu").numpy()
        # Plot_Raw_Data.plot_time_series_samples(average_value.squeeze(1).transpose(0, 2, 1), key_label='mean', num_samples=60, y_limit=(0, 0.4))
        # real_value = real.to("cpu").numpy()
        # Plot_Raw_Data.plot_time_series_samples(real_value.squeeze(1).transpose(0, 2, 1), key_label='real', num_samples=60, y_limit=(0, 0.4))
        # fake_value = generated.detach().to("cpu").numpy()
        # Plot_Raw_Data.plot_time_series_samples(fake_value.squeeze(1).transpose(0, 2, 1), key_label='fake', num_samples=60, y_limit=(0, 0.4))

        # Compute full batch reconstruction_loss
        if loss_fn == 'mse':
            reconstruction_loss = F.mse_loss(generated, avg_real_targets)
        elif loss_fn == 'l1':
            reconstruction_loss = F.l1_loss(generated, avg_real_targets)
        elif loss_fn == 'VGG19Loss':
            reconstruction_loss = self.perceptual_loss_calculator(generated, avg_real_targets)
        else:
            raise ValueError("Unsupported reconstruction_loss function.")

        return reconstruction_loss

    ## Calculates the gradient penalty loss for WGAN GP
    def compute_gradient_penalty(self, critic, real_samples, fake_samples, condition_label, device):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=device)
        # Get random interpolation between real and fake samples
        interpolates_C = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        # For conditional GANs, you might also want to interpolate conditioning factors if they vary per sample,
        d_interpolates = critic(interpolates_C, condition_label)

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
    def __init__(self, train_gan_data, transition_encoding, num_windows, window_length,
                 window_increment, window_shift, num_sample_per_condition, num_batches_per_epoch):
        """
        Args:
            train_gan_data: dict of {transition_name: {gen_data_1, gen_data_2, disc_data}}
            transition_encoding: dict mapping transition_name -> int label
            num_windows: number of time slices per sample
            window_length: how long each windowed segment is
            window_increment: step size to move between time windows
            window_shift: unused for now but reserved for future logic
            num_sample_per_condition: number of samples to draw per (transition, time_slice) per batch
            num_batches_per_epoch: how many batches per epoch
        """
        self.num_windows = num_windows
        self.window_shift = window_shift
        self.window_length = window_length
        self.window_increment = window_increment
        self.num_sample_per_condition = num_sample_per_condition

        # Store data grouped by transition
        self.data = []
        for transition_name, group in train_gan_data.items():
            label_index = transition_encoding[transition_name]
            self.data.append({
                'gen_data_1': [d.T for d in group['gen_data_1']],
                'gen_data_2': [d.T for d in group['gen_data_2']],
                'disc_data': [d.T for d in group['disc_data']],
                'transition_label': label_index
            })
        self.num_transitions = len(self.data)
        self.num_conditions = self.num_transitions * self.num_windows

        # Build sampling schedule: list of (transition_index, time_slice_index)
        self.sampling_schedule = []
        for _ in range(num_batches_per_epoch):
            for transition_index in range(self.num_transitions):
                for time_index in range(num_windows):
                    for _ in range(num_sample_per_condition):
                        self.sampling_schedule.append((transition_index, time_index))

    def __len__(self):
        return len(self.sampling_schedule)

    def __getitem__(self, idx):
        transition_index, time_slice = self.sampling_schedule[idx]
        group = self.data[transition_index]

        # Randomly pick a sample
        A_raw = group['gen_data_1'][np.random.randint(len(group['gen_data_1']))]
        B_raw = group['gen_data_2'][np.random.randint(len(group['gen_data_2']))]
        C_raw = group['disc_data'][np.random.randint(len(group['disc_data']))]

        # Apply Gaussian random shifts for slicing A, B, C separately
        base_start = self.window_shift + time_slice * self.window_increment  # starting from window_shift time point instead of 0
        shift_range = self.window_shift
        def sample_shift(std, max_shift):
            """Sample a truncated normal shift between [-max_shift, +max_shift]."""
            return int(truncnorm.rvs(-max_shift / std, max_shift / std, loc=0, scale=std))
        if shift_range > 0:
            time_shift = sample_shift(std=shift_range//2, max_shift=shift_range)
        else:
            time_shift = 0

        # Compute slice window
        start_A = base_start + time_shift
        start_B = base_start + time_shift
        start_C = base_start + time_shift
        end_A = start_A + self.window_length
        end_B = start_B + self.window_length
        end_C = start_C + self.window_length
        # Check bounds
        if end_A > A_raw.shape[1] or end_B > B_raw.shape[1] or end_C > C_raw.shape[1]:
            raise ValueError(f"Window exceeds sample bounds: A[{start_A}:{end_A}], B[{start_B}:{end_B}], C[{start_C}:{end_C}]")

        # Slice and convert to tensors
        A = torch.tensor(A_raw[:, start_A:end_A], dtype=torch.float32).unsqueeze(0)  # [1, C, T]
        B = torch.tensor(B_raw[:, start_B:end_B], dtype=torch.float32).unsqueeze(0)
        C_real = torch.tensor(C_raw[:, start_C:end_C], dtype=torch.float32).unsqueeze(0)
        condition_id = transition_index * self.num_windows + time_slice
        condition = torch.tensor(condition_id, dtype=torch.long)

        return A, B, C_real, condition



##  Define the VGG19 feature extractor for perceptual loss
class VGG19PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=None, use_input_norm=True, device='cuda'):
        super(VGG19PerceptualLoss, self).__init__()
        # Load pre-trained VGG19. We only need the features part.
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        # eval() mode ensures batchnorm and dropout layers are in evaluation mode

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            # Standard ImageNet normalization，If your EMG data is already in [0,1] and you want to apply this norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.transform = lambda x: (x - mean) / std  # If your EMG data is in a different range (e.g., -1 to 1 from Tanh), you might
            # first scale it to [0,1] then apply this transform, or adjust the transform.

        # Define which layers to extract features from VGG19 layers:
        if feature_layers is None:  # Default layers often used for perceptual loss
            self.feature_layers_indices = {'relu3_1': 11, 'relu4_1': 20, 'relu5_1': 29}
        else: # relu1_1: 1 (after conv1_1)，relu2_1: 6 (after conv2_1)， relu3_1: 11 (after conv3_1), relu4_1: 20 (after conv4_1)， relu5_1: 29 (after conv5_1)
            self.feature_layers_indices = feature_layers  # User can specify a dict like {'name': index}

        self.model_slices = nn.ModuleList()
        last_slice_idx = 0
        for name, layer_idx in sorted(self.feature_layers_indices.items(), key=lambda item: item[1]):
            self.model_slices.append(vgg19[last_slice_idx:layer_idx + 1])  # Grab up to and including the ReLU
            last_slice_idx = layer_idx + 1

        # Freeze VGG19 parameters - we don't want to train it
        for param in self.parameters():
            param.requires_grad = False
        self.loss_fn = nn.L1Loss()  # Or nn.MSELoss()

    def forward(self, generated_input, target_input):
        """
        Args:
            generated_input (torch.Tensor): Batch of generated EMG images, shape [B, 1, H, W]
            target_input (torch.Tensor): Batch of real EMG images, shape [B, 1, H, W]
        Returns:
            torch.Tensor: The perceptual loss value.
        """
        # 1. Replicate single channel to 3 channels for VGG
        generated_rgb = generated_input.repeat(1, 3, 1, 1)
        target_rgb = target_input.repeat(1, 3, 1, 1)

        # 2. Normalize if specified (assuming inputs are e.g. [0,1] or [-1,1] and transform handles it)
        if self.use_input_norm:
            generated_rgb = self.transform(generated_rgb)
            target_rgb = self.transform(target_rgb)

        # 3. Extract features and calculate loss
        perceptual_loss = 0.0
        current_gen_features = generated_rgb
        current_target_features = target_rgb

        for model_slice in self.model_slices:
            current_gen_features = model_slice(current_gen_features)
            current_target_features = model_slice(current_target_features)
            # Add loss for this layer's features
            perceptual_loss += self.loss_fn(current_gen_features, current_target_features)

        return perceptual_loss / len(self.model_slices)  # Average loss across selected layers


# --- Example Usage ---
if __name__ == '__main__':
    batch_size = 4
    img_height = 65
    img_width = 320
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy generated and real EMG data
    # Assuming data is roughly in [0, 1] range after some processing or Tanh output scaled
    generated_emg_data = torch.rand(batch_size, 1, img_height, img_width, device=device)
    real_emg_data = torch.rand(batch_size, 1, img_height, img_width, device=device)
    # In a real scenario, real_emg_data would come from your dataset
    # and generated_emg_data from your generator.
    # If your generator uses Tanh, its output is [-1, 1]. You'd scale it:
    # generated_emg_data_tanh = torch.tanh(torch.randn(batch_size, 1, img_height, img_width, device=device))
    # generated_emg_data = (generated_emg_data_tanh + 1) / 2.0 # Scale to [0, 1]

    # Initialize the perceptual loss module
    # You can specify which layers you want to use.
    # VGG19 layer names and approx indices for `features` module:
    # conv1_1 (0), relu1_1 (1)
    # conv2_1 (5), relu2_1 (6)
    # conv3_1 (10), relu3_1 (11)
    # conv4_1 (19), relu4_1 (20)
    # conv5_1 (28), relu5_1 (29)
    custom_feature_layers = {
        'relu2_1': 6,  # Earlier layer for texture/edges
        'relu4_1': 20  # Deeper layer for more complex patterns
    }
    perceptual_loss_calculator = VGG19PerceptualLoss(feature_layers=custom_feature_layers, device=device)
    # Or use default layers:
    # perceptual_loss_calculator = VGG19PerceptualLoss(device=device)


    # Calculate the perceptual loss
    loss_p = perceptual_loss_calculator(generated_emg_data, real_emg_data)
    print(f"Perceptual Loss: {loss_p.item()}")

    # This loss_p can then be added to your generator's total loss function
    # e.g., total_g_loss = adversarial_loss + lambda_perceptual * loss_p