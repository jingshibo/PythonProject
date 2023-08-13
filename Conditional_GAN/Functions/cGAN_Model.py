##
import torch
from torch import nn
from torchinfo import summary


## model structure
class ContractingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu', SN=False):
        super(ContractingBlock, self).__init__()
        if SN:  # spectral normalization
            self.conv1 = nn.utils.spectral_norm(
                nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, stride=1, padding=1, padding_mode='replicate'))
        else:
            self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, stride=1, padding=1,
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
    def __init__(self, input_channels, use_bn=True, SN=False):
        super(ExpandingBlock, self).__init__()
        if SN:  # spectral normalization
            self.conv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=1, padding=1))
        else:
            self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=1, padding=1)
        if use_bn:
            self.norm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.norm(x)
        x = self.activation(x)
        return x


class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels, SN=False):
        super(FeatureMapBlock, self).__init__()
        if SN:  # # spectral normalization
            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, padding_mode='replicate'))
        else:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x):
        x = self.conv(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_vector_dim, img_height, img_width, img_chan=1, hidden_channels=64):
        super(Generator, self).__init__()
        # data size
        self.input_dim = input_vector_dim
        self.img_chan = img_chan
        self.img_height = img_height
        self.img_width = img_width

        # dense layer to a suitable dim for reshaping
        self.fc = nn.Linear(input_vector_dim, img_chan * img_height * img_width)
        # encoder
        self.upfeature = FeatureMapBlock(img_chan, hidden_channels, SN=False)
        self.contract1 = ContractingBlock(hidden_channels, SN=False)
        self.contract2 = ContractingBlock(hidden_channels * 2, SN=False)
        # decoder
        self.expand2 = ExpandingBlock(hidden_channels * 4, SN=False)
        self.expand3 = ExpandingBlock(hidden_channels * 2, SN=False)
        self.downfeature = FeatureMapBlock(hidden_channels, img_chan, SN=False)
        self.sig = torch.nn.Sigmoid()

    def forward(self, noise_and_class):
        # convert the noise and class tensor of size (n_samples, input_dim) to a 4d ten sor of size (n_samples, input_dim, 1, 1)
        x = self.fc(noise_and_class)
        x = x.view(-1, self.img_chan, self.img_height, self.img_width)  # Reshaping to (batch_size, 2, 13, 10)

        x = self.upfeature(x)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.expand2(x)
        x = self.expand3(x)
        x = self.downfeature(x)
        xn = self.sig(x)
        return xn


## model summary
# model = Generator(81, 13, 10).to('cpu')  # move the model to GPU
# summary(model, input_size=(1024, 81))


class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()

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
        xn = self.final(x)
        return xn


## model summary
# model = Discriminator(18).to('cpu')  # move the model to GPU
# summary(model, input_size=(1024, 18, 13, 10))


## loss function
class LossFunction():
    def __init__(self, criterion):
        self.criterion = criterion

    def get_gen_loss(self, fake, image_one_hot_labels, disc):
        fake_image_and_labels = torch.cat((fake.float(), image_one_hot_labels.float()), 1)
        # error if you didn't concatenate your labels to your image correctly
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss

    def get_disc_loss(self, fake, real, image_one_hot_labels, disc):
        fake_image_and_labels = torch.cat((fake.float(), image_one_hot_labels.float()), 1)
        real_image_and_labels = torch.cat((real.float(), image_one_hot_labels.float()), 1)
        disc_fake_pred = disc(fake_image_and_labels.detach())
        disc_real_pred = disc(real_image_and_labels)

        disc_fake_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        return disc_loss
