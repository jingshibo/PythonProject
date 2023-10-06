import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import datetime
import numpy as np
from Conditional_GAN.Models import cGAN_Model, cGAN_Loss, Model_Storage, cGAN_DataSet

## training process
class ModelTraining():
    def __init__(self, num_epochs, batch_size, sampling_repetition, gen_update_interval, disc_update_interval, decay_epochs, noise_dim, blending_factor_dim):
        #  initialize member variables
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = f'/Conditional_GAN/Others\\runs_{timestamp}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.sampling_repetition = sampling_repetition
        self.gen_update_interval = gen_update_interval
        self.disc_update_interval = disc_update_interval
        self.noise_dim = noise_dim
        self.blending_factor_dim = blending_factor_dim
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
        self.writer = None

    def trainModel(self, train_data, checkpoint_model_path, checkpoint_result_path, training_parameters, transition_type, select_channels='emg_all'):
        # input data
        dataset = cGAN_DataSet.RandomMixEmgDataSet(train_data, self.batch_size, self.sampling_repetition)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0)
        self.img_channel = train_data['gen_data_1']['timepoint_0'][0].shape[1]
        self.img_height = train_data['gen_data_1']['timepoint_0'][0].shape[2]
        self.img_width = train_data['gen_data_1']['timepoint_0'][0].shape[3]
        self.n_classes = len(list(train_data['gen_data_1'].keys()))
        self.display_step = int(len(dataset) / self.batch_size)

        # training model
        generator_input_dim = self.noise_dim + self.n_classes
        discriminator_input_channel = self.img_channel + self.n_classes
        self.gen = cGAN_Model.Generator_UNet(generator_input_dim, self.img_height, self.img_width, self.blending_factor_dim, self.n_classes).to(self.device)
        self.disc = cGAN_Model.Discriminator_Same(discriminator_input_channel, self.n_classes).to(self.device)

        # training parameters
        gen_lr = 0.0003  # initial learning rate
        disc_lr = 0.0002
        gen_lr_decay_rate = 0.7
        disc_lr_decay_rate = 0.7
        weight_decay = 0.0000
        beta = (0.7, 0.999)

        # optimizer
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=gen_lr, weight_decay=weight_decay, betas=beta)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=disc_lr, weight_decay=weight_decay, betas=beta)

        # learning rate scheduler
        # decay_steps = self.decay_epochs * len(self.train_loader)  # take the repetition of gen into account
        # self.lr_gen_opt = torch.optim.lr_scheduler.StepLR(self.gen_opt, step_size=decay_steps // self.gen_update_interval,
        #     gamma=gen_lr_decay_rate)
        # self.lr_disc_opt = torch.optim.lr_scheduler.StepLR(self.disc_opt, step_size=decay_steps // self.disc_update_interval,
        #     gamma=disc_lr_decay_rate)
        self.lr_gen_opt = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.gen_opt, milestones=self.decay_epochs, gamma=gen_lr_decay_rate)
        self.lr_disc_opt = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.disc_opt, milestones=self.decay_epochs, gamma=disc_lr_decay_rate)

        # loss function
        # criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.MSELoss()
        # self.loss_fn = cGAN_Loss.LossFunction(criterion)
        c_lambda = 10  # the weight of the gradient penalty
        var_weight = 0.003  # the weight of blending factor variance
        construct_weight = 0.7  # the weight of constructed critic value
        factor_1_weight = 0  # the weight of blending factor 1
        factor_2_weight = 0  # the weight of blending factor 2
        factor_3_weight = 0.001  # the weight of blending factor 2
        self.loss_fn = cGAN_Loss.WGANloss(c_lambda, var_weight, construct_weight, factor_1_weight, factor_2_weight, factor_3_weight)

        # train the model
        models = {'gen': self.gen, 'disc': self.disc}

        for epoch_number in range(self.num_epochs):  # loop over each epoch
            # train the model
            self.trainOneEpoch(epoch_number)
            self.lr_disc_opt.step()  # update the learning rate
            self.lr_gen_opt.step()  # update the learning rate

            # set the checkpoints to save models
            if (epoch_number + 1) % 10 == 0 and (epoch_number + 1) >= 10:
                print(f"Saved checkpoint at epoch {epoch_number + 1}")
                Model_Storage.saveCheckPointModels(checkpoint_model_path, epoch_number + 1, models, transition_type)
                # estimate blending factors
                blending_factors = self.estimateBlendingFactors(dataset, train_data)
                Model_Storage.saveCheckPointCGanResults(checkpoint_result_path, epoch_number + 1, blending_factors, training_parameters, transition_type)

        # estimate blending factors
        blending_factors = self.estimateBlendingFactors(dataset, train_data)
        return models, blending_factors

    def trainOneEpoch(self, epoch_number):
        self.gen.train(True)
        self.disc.train(True)

        batch_count = 0  # Counter to keep track of the number of batches
        for gen_data, disc_data in tqdm(self.train_loader):
            # image_width = image.shape [3]

            batch_count += 1  # first train discriminator multiple times, then start to train generator
            cur_batch_size = len(disc_data)
            condition = gen_data[0].to(self.device)
            gen_data_1 = gen_data[1].to(self.device)
            gen_data_2 = gen_data[2].to(self.device)
            real = disc_data.to(self.device)

            # convert condition into one hot vectors
            one_hot_labels = F.one_hot(condition, self.n_classes)
            # Apply one-side label smoothing
            # self.epsilon = 0.90
            # one_hot_labels = one_hot_labels * self.epsilon
            # adding two additional dimensions to the one-hot encoded labels (size [batch_size, n_classes, 1, 1])
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            # match the spatial dimensions of the image.(size [batch_size, n_classes, image_height, image_width])
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, self.img_height, self.img_width)

            # Update the discriminator every m batches
            if batch_count % self.disc_update_interval == 0:
                self.disc_opt.zero_grad()  # Zero out the discriminator gradients
                fake, blending_factors = self.generateFakeData(cur_batch_size, one_hot_labels, gen_data_1, gen_data_2)
                disc_loss = self.loss_fn.get_disc_loss(fake, real, image_one_hot_labels, one_hot_labels, self.disc)
                disc_loss.backward()  # Update gradients
                self.disc_opt.step()  # Update optimizer
                self.disc_losses += [disc_loss.item()]

            # Update the generator every n batches
            if batch_count % self.gen_update_interval == 0:
                self.gen_opt.zero_grad()
                fake, blending_factors = self.generateFakeData(cur_batch_size, one_hot_labels, gen_data_1, gen_data_2)
                gen_loss = self.loss_fn.get_gen_loss(fake, real, blending_factors, image_one_hot_labels, one_hot_labels, self.disc)
                gen_loss.backward()
                self.gen_opt.step()
                self.gen_losses += [gen_loss.item()]

            if (self.current_step + 1) % self.display_step == 0:
                disc_mean_loss = sum(self.disc_losses[-(self.display_step // self.disc_update_interval):]) / (
                        self.display_step // self.disc_update_interval)
                gen_mean_loss = sum(self.gen_losses[-(self.display_step // self.gen_update_interval):]) / (
                        self.display_step // self.gen_update_interval)
                print(f"Epoch {epoch_number + 1}: Step {self.current_step + 1}: Generator loss: {gen_mean_loss}, Discriminator loss: "
                      f"{disc_mean_loss}, gen_lr: {self.lr_gen_opt.get_last_lr()}, disc_lr: {self.lr_disc_opt.get_last_lr()}")
                # print(f"Epoch {epoch_number}: Step {self.current_step}: Generator loss: {gen_mean_loss}, Discriminator loss: "
                #       f"{disc_mean_loss}, learning_rate: 0.0002")
            self.current_step += 1

    def generateFakeData(self, cur_batch_size, one_hot_labels, gen_data_1, gen_data_2):
        # estimate blending factors
        if self.noise_dim > 0:
            # Get noise corresponding to the current batch_size
            fake_noise = torch.randn(cur_batch_size, self.noise_dim, device=self.device)
            # Combine the noise vectors and the one-hot labels for the generator
            noise_and_labels = torch.cat((fake_noise.float(), one_hot_labels.float()), 1)
        else:
            noise_and_labels = one_hot_labels.float()
        blending_factors = self.gen(noise_and_labels, one_hot_labels)  # Estimate blending factors

        # Generate fake images, according to the size of blending factors
        if self.blending_factor_dim == 1:
            fake = blending_factors * gen_data_1 + (1 - blending_factors) * gen_data_2  # Generate the conditioned fake images
        elif self.blending_factor_dim == 2:
            fake = blending_factors[:, 0, :, :].unsqueeze(1) * gen_data_1 + blending_factors[:, 1, :, :].unsqueeze(1) * gen_data_2
        elif self.blending_factor_dim == 3:
            fake = blending_factors[:, 0, :, :].unsqueeze(1) * gen_data_1 + blending_factors[:, 1, :, :].unsqueeze(
                1) * gen_data_2 + blending_factors[:, 2, :, :].unsqueeze(1)
        return fake, blending_factors

    def estimateBlendingFactors(self, dataset, train_data):
        blending_factors = {}
        n_iterations = 1000  # Number of times to calculate blending_factor at each time_point for averaging purpose

        self.gen.train(False)  # Set the generator to evaluation mode
        with torch.no_grad():  # Disable autograd for better performance
            for time_point in list(train_data['gen_data_1'].keys()):
                number = dataset.extract_and_normalize(time_point)
                one_hot_labels = F.one_hot(torch.tensor([number] * n_iterations), self.n_classes).to(self.device)
                # one_hot_labels = self.epsilon * F.one_hot(torch.tensor([number] * n_iterations), self.n_classes).to(self.device)

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
                mean_blending_factor = np.mean(blending_outputs, axis=0, keepdims=True)
                # Store the average blending_factor
                blending_factors[time_point] = mean_blending_factor
        return blending_factors

## In order to train multiple transition types, we wrap up a single type of training process into a function
def trainCGan(train_gan_data, transition_type, training_parameters, storage_parameters):
    # Extract values from the input dictionaries
    num_epochs = training_parameters['num_epochs']
    batch_size = training_parameters['batch_size']
    sampling_repetition = training_parameters['sampling_repetition']
    gen_update_interval = training_parameters['gen_update_interval']
    disc_update_interval = training_parameters['disc_update_interval']
    decay_epochs = training_parameters['decay_epochs']
    noise_dim = training_parameters['noise_dim']
    blending_factor_dim = training_parameters['blending_factor_dim']

    checkpoint_model_path = storage_parameters['checkpoint_model_path']
    checkpoint_result_path = storage_parameters['checkpoint_result_path']
    subject = storage_parameters['subject']
    version = storage_parameters['version']
    model_type = storage_parameters['model_type']
    model_name = storage_parameters['model_name']
    result_set = storage_parameters['result_set']

    # train gan model
    now = datetime.datetime.now()
    train_model = ModelTraining(num_epochs, batch_size, sampling_repetition, gen_update_interval, disc_update_interval, decay_epochs,
        noise_dim, blending_factor_dim)
    gan_models, blending_factors = train_model.trainModel(train_gan_data, checkpoint_model_path, checkpoint_result_path,
        training_parameters, transition_type)
    print(datetime.datetime.now() - now)

    # save trained gan models and results
    Model_Storage.saveModels(gan_models, subject, version, model_type, model_name, transition_type=transition_type, project='cGAN_Model')
    # save model results
    Model_Storage.saveCGanResults(subject, blending_factors, version, result_set, training_parameters, model_type,
        transition_type=transition_type, project='cGAN_Model')

    return gan_models, blending_factors

