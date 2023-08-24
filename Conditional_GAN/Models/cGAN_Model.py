##
import torch
from torch import nn
from torchinfo import summary


## model structure
class ContractingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True, stride=1, padding=1, activation='relu', SN=False):
        super(ContractingBlock, self).__init__()
        if SN:  # spectral normalization
            self.conv1 = nn.utils.spectral_norm(
                nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, stride=stride, padding=padding, padding_mode='replicate'))
        else:
            self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, stride=stride, padding=padding,
                padding_mode='replicate')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.norm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.norm(x)
        x = self.activation(x)
        return x


class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True, stride=1, padding=1, SN=False):
        super(ExpandingBlock, self).__init__()
        if SN:  # spectral normalization
            self.conv1 = nn.utils.spectral_norm(
                nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=stride, padding=padding))
        else:
            self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=stride, padding=padding)
        if use_bn:
            self.norm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

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
    def __init__(self, input_vector_dim, img_height, img_width, output_chan, hidden_channels=64):
        super(Generator_UNet, self).__init__()
        # data size
        self.input_dim = input_vector_dim
        self.output_chan = output_chan
        self.img_height = img_height
        self.img_width = img_width

        # dense layer to a suitable dim for reshaping
        self.fc = nn.Linear(input_vector_dim, output_chan * img_height * img_width)

        # encoder
        self.upfeature = FeatureMapBlock(output_chan, hidden_channels, stride=1, SN=False)
        self.contract1 = ContractingBlock(hidden_channels, stride=1, SN=False)
        self.contract2 = ContractingBlock(hidden_channels * 2, stride=1, SN=False)

        # decoder
        self.expand2 = ExpandingBlock(hidden_channels * 4 + hidden_channels * 2, stride=1, SN=False)  # Adjusted for skip connection
        self.expand3 = ExpandingBlock(hidden_channels * 3 + hidden_channels, stride=1, SN=False)  # Adjusted for skip connection
        self.downfeature = FeatureMapBlock(hidden_channels * 2, output_chan, stride=1, SN=False)

        self.sig = torch.nn.Sigmoid()

    def forward(self, noise_and_class):
        # convert the noise and class tensor of size (n_samples, input_dim) to a 4d tensor of size (n_samples, input_dim, 1, 1)
        x = self.fc(noise_and_class)
        x = x.view(-1, self.output_chan, self.img_height, self.img_width)  # Reshaping to (batch_size, 2, 13, 10)

        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)

        x3 = self.expand2(x2, x1)  # Skip connection from contract1 to expand2
        x4 = self.expand3(x3, x0)  # Skip connection from upfeature to expand3

        x5 = self.downfeature(x4)
        x6 = self.sig(x5)
        return x6


## model summary
# model = Generator_UNet(81, 13, 10, 2).to('cpu')  # move the model to GPU
# summary(model, input_size=(1024, 81))


##
class Discriminator_Same(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator_Same, self).__init__()

        self.upfeature = FeatureMapBlock(input_channels, hidden_channels, SN=True)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, activation='lrelu', SN=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, activation='lrelu', SN=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, activation='lrelu', SN=True)
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, image_and_class):
        x = self.upfeature(image_and_class)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.contract3(x)
        x = self.final(x)
        return x

##
class Discriminator_Shrinking(nn.Module):
    def __init__(self, input_channels, padding='valid', hidden_channels=64):
        super(Discriminator_Shrinking, self).__init__()

        self.upfeature = FeatureMapBlock(input_channels, hidden_channels, padding=padding, SN=True)
        self.contract1 = ContractingBlock(hidden_channels, padding=padding, use_bn=False, activation='lrelu', SN=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, padding=padding, activation='lrelu', SN=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, padding=padding, activation='lrelu', SN=True)
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, image_and_class):
        x = self.upfeature(image_and_class)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.contract3(x)
        x = self.final(x)
        return x


## model summary
# model = Discriminator_Shrinking(1).to('cpu')  # move the model to GPU
# summary(model, input_size=(1024, 1, 13, 10))


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
    def __init__(self, c_lambda):
        self.c_lambda = c_lambda  # the current weight of the gradient penalty

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

    # Return the loss of a generator given the discriminator's scores of the generator's fake images.
    def get_gen_loss(self, fake, image_one_hot_labels, disc):
        fake_image_and_labels = torch.cat((fake.float(), image_one_hot_labels.float()), 1)
        disc_fake_pred = disc(fake_image_and_labels)  # error if you didn't concatenate your labels to your image correctly
        gen_loss = -1. * torch.mean(disc_fake_pred)
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
        # disc_loss = torch.mean((disc_fake_pred-disc_real_pred)**2) + self.c_lambda * g_p

        return disc_loss



##
class ModifiedGenerator(nn.Module):
    '''
    Modified Generator Class
    '''
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(ModifiedGenerator, self).__init__()
        self.input_dim = input_dim

        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4, kernel_size=3, stride=2),  # First layer, starts with 1x1
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=1, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)


model = ModifiedGenerator(20).to('cpu')  # move the model to GPU
summary(model, input_size=(128, 20))