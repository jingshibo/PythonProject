##
import torch
from torch import nn
import torch.nn.functional as F


##
class Generator(nn.Module):
    def __init__(self, input_dim=10, img_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, img_chan, kernel_size=4, final_layer=True),
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
                nn.Tanh(),
            )

    def forward(self, noise):
        # noise: a noise tensor with dimensions (n_samples, input_dim)
        x = noise.view(len(noise), self.input_dim, 1, 1)
        x = gen(x)
        return x


##
class Discriminator(nn.Module):
    def __init__(self, img_chan=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(img_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        # Given an image tensor, returns a 1-dimension tensor representing fake/real.
        x = self.disc(image)
        x = x.view(len(x), -1)
        return x


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

