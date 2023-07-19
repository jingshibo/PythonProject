import torch
from torch import nn


## model structure
class ResidualBlock(nn.Module):
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x

class ContractingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, stride=1, padding=1, padding_mode='replicate')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=1, padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x):
        x = self.conv(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.expand2(x4)
        x6 = self.expand3(x5)
        xn = self.downfeature(x6)
        return self.tanh(xn)

class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn


## loss function
class LossFunction():
    def __init__(self, adv_criterion, recon_criterion):
        self.adv_criterion = adv_criterion
        self.identity_criterion = recon_criterion
        self.cycle_criterion = recon_criterion

    def get_gen_adversarial_loss(self, real_X, disc_Y, gen_XY):
        fake_Y = gen_XY(real_X)
        disc_fake_Y_hat = disc_Y(fake_Y)
        adversarial_loss = self.adv_criterion(disc_fake_Y_hat, torch.ones_like(disc_fake_Y_hat))
        return adversarial_loss, fake_Y

    def get_identity_loss(self, real_X, gen_YX):
        identity_X = gen_YX(real_X)
        identity_loss = self.identity_criterion(identity_X, real_X)
        return identity_loss, identity_X

    def get_cycle_consistency_loss(self, real_X, fake_Y, gen_YX):
        cycle_X = gen_YX(fake_Y)
        cycle_loss = self.cycle_criterion(cycle_X, real_X)
        return cycle_loss, cycle_X

    def get_gen_loss(self, real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, lambda_identity=0.1, lambda_cycle=10):
        # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
        adv_loss_BA, fake_A = self.get_gen_adversarial_loss(real_B, disc_A, gen_BA)
        adv_loss_AB, fake_B = self.get_gen_adversarial_loss(real_A, disc_B, gen_AB)
        gen_adversarial_loss = adv_loss_BA + adv_loss_AB

        # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
        identity_loss_A, identity_A = self.get_identity_loss(real_A, gen_BA)
        identity_loss_B, identity_B = self.get_identity_loss(real_B, gen_AB)
        gen_identity_loss = identity_loss_A + identity_loss_B

        # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
        cycle_loss_BA, cycle_A = self.get_cycle_consistency_loss(real_A, fake_B, gen_BA)
        cycle_loss_AB, cycle_B = self.get_cycle_consistency_loss(real_B, fake_A, gen_AB)
        gen_cycle_loss = cycle_loss_BA + cycle_loss_AB

        # Total loss
        gen_loss = gen_adversarial_loss + lambda_identity * gen_identity_loss + lambda_cycle * gen_cycle_loss
        return gen_loss

    def get_disc_loss(self, real_X, fake_X, disc_X):
        disc_fake_X_hat = disc_X(fake_X.detach())  # Detach generator
        disc_fake_X_loss = self.adv_criterion(disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
        disc_real_X_hat = disc_X(real_X)
        disc_real_X_loss = self.adv_criterion(disc_real_X_hat, torch.ones_like(disc_real_X_hat))
        disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2
        return disc_loss


