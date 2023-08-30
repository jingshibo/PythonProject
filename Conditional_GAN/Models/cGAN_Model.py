##
import torch
from torch import nn
from torchinfo import summary


## model structure
class ContractingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True, stride=1, kernel_size=3, padding=1, activation='relu', SN=False):
        super(ContractingBlock, self).__init__()
        if SN:  # spectral normalization
            self.conv1 = nn.utils.spectral_norm(
                nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate'))
        else:
            self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.norm = nn.InstanceNorm2d(input_channels * 2)
            # self.norm = nn.LayerNorm([input_channels * 2, 13, 10])
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.norm(x)
        x = self.activation(x)
        return x


class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True, stride=1, kernel_size=3, padding=1, activation='relu', SN=False):
        super(ExpandingBlock, self).__init__()
        if SN:  # spectral normalization
            self.conv1 = nn.utils.spectral_norm(
                nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=kernel_size, stride=stride, padding=padding))
        else:
            self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.norm = nn.InstanceNorm2d(input_channels // 2)
            # self.norm = nn.LayerNorm([input_channels // 2, 13, 10])
        self.use_bn = use_bn

    def forward(self, x, skip_con):
        x = torch.cat([x, skip_con], 1)
        x = self.conv1(x)
        if self.use_bn:
            x = self.norm(x)
        x = self.activation(x)
        return x


class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, padding=1, SN=False):
        super(FeatureMapBlock, self).__init__()
        if SN:  # # spectral normalization
            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=padding, padding_mode='replicate'))
        else:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=padding, padding_mode='replicate')

    def forward(self, x):
        x = self.conv(x)
        return x


# UNet Generator
class Generator_UNet(nn.Module):
    def __init__(self, input_vector_dim, img_height, img_width, output_chan, hidden_channels=32):
        super(Generator_UNet, self).__init__()
        # data size
        self.input_dim = input_vector_dim
        self.output_chan = output_chan
        self.img_height = img_height
        self.img_width = img_width

        # dense layer to a suitable dim for reshaping
        self.fc = nn.Linear(input_vector_dim, output_chan * img_height * img_width)

        # encoder
        self.upfeature = FeatureMapBlock(output_chan, hidden_channels, stride=1, SN=True)
        self.contract1 = ContractingBlock(hidden_channels, kernel_size=3, padding=1, SN=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=3, padding=1, SN=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=3, padding=1, SN=True)
        self.contract4 = ContractingBlock(hidden_channels * 8, kernel_size=3, padding=1, SN=True)

        # decoder
        self.expand4 = ExpandingBlock(hidden_channels * 16 + hidden_channels * 8, kernel_size=3, padding=1, SN=True)
        self.expand3 = ExpandingBlock(hidden_channels * 12 + hidden_channels * 4, kernel_size=3, padding=1, SN=True)
        self.expand2 = ExpandingBlock(hidden_channels * 8 + hidden_channels * 2, kernel_size=3, padding=1, SN=True)
        self.expand1 = ExpandingBlock(hidden_channels * 5 + hidden_channels, kernel_size=3, padding=1, SN=True)
        self.downfeature = FeatureMapBlock(int(hidden_channels * 3), output_chan, stride=1, SN=True)

        self.sig = torch.nn.Sigmoid()

    def forward(self, noise_and_class):
        # convert the noise and class tensor of size (n_samples, input_dim) to a 4d tensor of size (n_samples, input_dim, 1, 1)
        x = self.fc(noise_and_class)
        x = x.view(-1, self.output_chan, self.img_height, self.img_width)  # Reshaping to (batch_size, 2, 13, 10)

        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        y4 = self.expand4(x4, x3)  # Skip connection from contract1 to expand2
        y3 = self.expand3(y4, x2)  # Skip connection from contract1 to expand2
        y2 = self.expand2(y3, x1)  # Skip connection from upfeature to expand3
        y1 = self.expand1(y2, x0)  # Skip connection from upfeature to expand3
        y0 = self.downfeature(y1)
        yn = self.sig(y0)
        return yn


## model summary
# model = Generator_UNet(81, 13, 10, 2).to('cpu')  # move the model to GPU
# summary(model, input_size=(64, 81))


##
class Discriminator_Same(nn.Module):
    def __init__(self, input_channels, hidden_channels=32):
        super(Discriminator_Same, self).__init__()

        self.upfeature = FeatureMapBlock(input_channels, hidden_channels, SN=True)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=3, padding=1, activation='lrelu', SN=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=3, padding=1, activation='lrelu', SN=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=3, padding=1, activation='lrelu', SN=True)
        self.contract4 = ContractingBlock(hidden_channels * 8, kernel_size=3, padding=1, activation='lrelu', SN=True)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, image_and_class):
        x = self.upfeature(image_and_class)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.contract3(x)
        x = self.contract4(x)
        x = self.final(x)
        return x

class Discriminator_Shrinking(nn.Module):
    def __init__(self, input_channels, padding='valid', hidden_channels=32):
        super(Discriminator_Shrinking, self).__init__()

        self.upfeature = FeatureMapBlock(input_channels, hidden_channels, SN=True)
        self.contract1 = ContractingBlock(hidden_channels, padding=padding, use_bn=False, activation='lrelu', SN=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, padding=padding, activation='lrelu', SN=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, padding=padding, activation='lrelu', SN=True)
        self.contract4 = ContractingBlock(hidden_channels * 8, padding=padding, activation='lrelu', SN=True)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, image_and_class):
        x = self.upfeature(image_and_class)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.contract3(x)
        x = self.contract4(x)
        x = self.final(x)
        return x


## model summary
# model = Discriminator_Same(1).to('cpu')  # move the model to GPU
# summary(model, input_size=(64, 1, 13, 10))


## loss function
class LossFunction():
    def __init__(self, criterion):
        self.discerion = criterion

    def get_gen_loss(self, fake, image_one_hot_labels, disc):
        fake_image_and_labels = torch.cat((fake.float(), image_one_hot_labels.float()), 1)
        # error if you didn't concatenate your labels to your image correctly
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = self.discerion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss

    def get_disc_loss(self, fake, real, image_one_hot_labels, disc):
        fake_image_and_labels = torch.cat((fake.float(), image_one_hot_labels.float()), 1)
        real_image_and_labels = torch.cat((real.float(), image_one_hot_labels.float()), 1)
        disc_fake_pred = disc(fake_image_and_labels.detach())
        disc_real_pred = disc(real_image_and_labels)

        disc_fake_loss = self.discerion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = self.discerion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        return disc_loss


## GRADED FUNCTION: get_gradient
class WGANloss():
    def __init__(self, c_lambda, var_weight, construct_weight):
        self.c_lambda = c_lambda  # the weight of the gradient penalty
        self.var_weight = var_weight  # the weight of blending factor variance
        self.construct_weight = construct_weight  # the weight of blending factor variance

    # Return the gradient of the discic's scores with respect to mixes of real and fake images.
    def get_gradient(self, disc, real, fake, epsilon):
        # Get mixed images
        mixed_images = real * epsilon + fake * (1 - epsilon)
        # Calculate the discic's scores on the mixed images
        mixed_scores = disc(mixed_images)
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            # Note: You need to take the gradient of outputs with respect to inputs.
            inputs=mixed_images,
            outputs=mixed_scores,
            # These other parameters have to do with the pytorch autograd engine works
            grad_outputs=torch.ones_like(mixed_scores), create_graph=True, retain_graph=True)[0]
        return gradient

    # Return the gradient penalty, given a gradient.
    def gradient_penalty(self, gradient):
        # Flatten the gradients so that each row captures one image
        gradient = gradient.view(len(gradient), -1)
        # Calculate the magnitude of every row
        gradient_norm = gradient.norm(2, dim=1)
        # Penalize the mean squared distance of the gradient norms from 1
        penalty = torch.mean((gradient_norm - 1) ** 2)
        return penalty

    # computes the average absolute difference between the disc's prediction values for fake and real images (for each class separately)
    def getConstructDifference(self, image_one_hot_labels, disc_fake_pred, disc_real_pred):
        # Initialize variables to store the summed class-specific mean differences and the count of non-empty classes
        class_specific_mean_diff_sum = 0.0
        non_empty_class_count = 0
        n_classes = image_one_hot_labels.shape[1]

        # Loop through each class to calculate class-specific mean differences
        for i in range(n_classes):
            #  # Create a boolean mask for samples belonging to the i-th class
            class_mask = image_one_hot_labels[:, i, 0, 0].bool()  # get a vector of boolean mask (set True for samples belonging to the ith class.)

            # Use the mask to get critic values for fake and real images of the ith class
            class_fake_pred = disc_fake_pred[class_mask]  # uses the boolean mask to extract the fake image critic values of the ith class.
            class_real_pred = disc_real_pred[class_mask]  # uses the boolean mask to extract the real image critic values of the ith class.

            if class_fake_pred.numel() > 0 and class_real_pred.numel() > 0:  # Check if there are samples for this class in the batch
                class_specific_mean_diff = torch.abs(torch.mean(class_fake_pred) - torch.mean(class_real_pred))
                class_specific_mean_diff_sum += class_specific_mean_diff
                non_empty_class_count += 1

        # Calculate the mean of class-specific mean differences
        if non_empty_class_count > 0:
            average_class_specific_diff = class_specific_mean_diff_sum / non_empty_class_count
        else:
            average_class_specific_diff = 0.0
        return average_class_specific_diff

    # Compute the variance value of blending_factor along the height and width dimensions for each sample, treat each channel separately
    def getVarianceValue(self, blending_factor):
        num_channels = blending_factor.shape[1]  # get the number of channels
        # Initialize a tensor to store variance terms for each channel
        combined_variance_term = torch.zeros(blending_factor.shape[0]).to(blending_factor.device)

        # Compute the variance along the height and width dimensions of each channel separately for each sample
        for channel in range(num_channels):
            variance_term = torch.var(blending_factor[:, channel, :, :], dim=[1, 2])
            combined_variance_term += variance_term  # combine the variance of each sample
        mean_variance_value = torch.mean(combined_variance_term)
        return mean_variance_value

    def get_gen_loss(self, fake, real, blending_factor, image_one_hot_labels, disc):
        fake_image_and_labels = torch.cat((fake.float(), image_one_hot_labels.float()), 1)
        real_image_and_labels = torch.cat((real.float(), image_one_hot_labels.float()), 1)
        disc_fake_pred = disc(fake_image_and_labels)
        disc_real_pred = disc(real_image_and_labels)

        # Compute the variance value of blending_factor along the height and width dimensions for each sample, treat each channel separately
        variance_value = self.getVarianceValue(blending_factor)
        # Compute the construct error between the generated data and real data in order to minimize the difference per class
        construct_error = self.getConstructDifference(image_one_hot_labels, disc_fake_pred, disc_real_pred)
        # construct_error = torch.abs((torch.mean(disc_fake_pred) - torch.mean(disc_real_pred)))

        # Add the variance term and construct error to the loss
        gen_loss = -1. * torch.mean(disc_fake_pred) + self.var_weight * variance_value + self.construct_weight * construct_error
        return gen_loss

    # Return the loss of a disc given the disc's scores for fake and real images, the gradient penalty, and gradient penalty weight.
    def get_disc_loss(self, fake, real, image_one_hot_labels, disc):
        fake_image_and_labels = torch.cat((fake.float(), image_one_hot_labels.float()), 1)
        real_image_and_labels = torch.cat((real.float(), image_one_hot_labels.float()), 1)
        disc_fake_pred = disc(fake_image_and_labels.detach())
        disc_real_pred = disc(real_image_and_labels)

        epsilon = torch.rand(len(real), 1, 1, 1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=True)
        gradient = self.get_gradient(disc, real_image_and_labels, fake_image_and_labels.detach(), epsilon)
        g_p = self.gradient_penalty(gradient)

        disc_loss = torch.mean(disc_fake_pred) - torch.mean(disc_real_pred) + self.c_lambda * g_p  # torch.mean() averages all value in a tensor
        return disc_loss


##
class ModifiedGenerator(nn.Module):
    '''
    Modified Generator Class
    '''
    def __init__(self, input_vector_dim, img_height, img_width, output_chan, hidden_channels=32):
        super(ModifiedGenerator, self).__init__()
        # data size
        self.input_dim = input_vector_dim
        self.output_chan = output_chan
        self.img_height = img_height
        self.img_width = img_width

        # dense layer to a suitable dim for reshaping
        self.fc = nn.Linear(input_vector_dim, output_chan * img_height * img_width)

        self.upfeature = FeatureMapBlock(output_chan, hidden_channels, SN=True)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, activation='lrelu', SN=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, activation='lrelu', SN=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, activation='lrelu', SN=True)
        self.contract4 = ContractingBlock(hidden_channels * 8, activation='lrelu', SN=True)
        self.final = nn.Conv2d(hidden_channels * 16, output_chan, kernel_size=1)

        self.sig = torch.nn.Sigmoid()

    def forward(self, noise_and_class):
        x = self.fc(noise_and_class)
        x = x.view(-1, self.output_chan, self.img_height, self.img_width)  # Reshaping to (batch_size, 2, 13, 10)
        x = self.upfeature(x)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.contract3(x)
        x = self.contract4(x)
        x = self.final(x)
        x = self.sig(x)

        return x

# model = ModifiedGenerator(20).to('cpu')  # move the model to GPU
# summary(model, input_size=(128, 20))