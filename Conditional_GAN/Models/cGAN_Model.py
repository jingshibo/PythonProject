##
import torch
from torch import nn
from torchinfo import summary


## model structure
class ConditionalNorm2d(nn.Module):
    def __init__(self, num_channels, num_classes, norm='batch_norm'):
        super().__init__()
        self.num_channels = num_channels
        if norm == 'batch_norm':
            self.norm = nn.BatchNorm2d(num_channels, affine=False)
        elif norm == 'instance_norm':
            self.norm = nn.InstanceNorm2d(num_channels, affine=False)
        self.embed = nn.Embedding(num_classes, num_channels * 2)
        self.embed.weight.data[:, :num_channels].fill_(1.)  # Initialize scale at 1
        self.embed.weight.data[:, num_channels:].zero_()  # Initialize bias at 0
        # self.embed = nn.Linear(num_classes, num_channels * 2)

    def forward(self, x, label):
        out = self.norm(x)
        # Convert one_hot to index tensor
        label = torch.argmax(label, dim=1)
        gamma, beta = self.embed(label.to(torch.long)).chunk(2, 1)
        # gamma, beta = self.embed(label.to(torch.float32)).chunk(2, 1)
        gamma = gamma.view(-1, self.num_channels, 1, 1)
        beta = beta.view(-1, self.num_channels, 1, 1)
        out = gamma * out + beta
        return out

class ContractingBlock(nn.Module):
    def __init__(self, input_channels, num_classes, use_norm=True, stride=1, kernel_size=3, padding=1, activation='relu', SN=False):
        super(ContractingBlock, self).__init__()
        if SN:  # spectral normalization
            self.conv1 = nn.utils.spectral_norm(
                nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate'))
        else:
            self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_norm:
            self.norm = ConditionalNorm2d(input_channels * 2, num_classes, norm='batch_norm')
        self.use_norm = use_norm

    def forward(self, x, label):
        x = self.conv1(x)
        if self.use_norm:
            x = self.norm(x, label)
        x = self.activation(x)
        return x


class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, num_classes, use_norm=True, stride=1, kernel_size=3, padding=1, activation='relu', SN=False):
        super(ExpandingBlock, self).__init__()
        if SN:  # spectral normalization
            self.conv1 = nn.utils.spectral_norm(
                nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=kernel_size, stride=stride, padding=padding))
        else:
            self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_norm:
            self.norm = ConditionalNorm2d(input_channels // 2, num_classes, norm='batch_norm')
        self.use_norm = use_norm

    def forward(self, x, skip_con, label):
        x = torch.cat([x, skip_con], 1)
        x = self.conv1(x)
        if self.use_norm:
            x = self.norm(x, label)
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
    def __init__(self, input_vector_dim, img_height, img_width, output_chan, num_classes, hidden_channels=32):
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
        self.contract1 = ContractingBlock(hidden_channels, num_classes, kernel_size=3, padding=1, activation='relu', SN=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, num_classes, kernel_size=3, padding=1, activation='relu', SN=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, num_classes, kernel_size=3, padding=1, activation='relu', SN=True)
        self.contract4 = ContractingBlock(hidden_channels * 8, num_classes, kernel_size=3, padding=1, activation='relu', SN=True)

        # decoder
        self.expand4 = ExpandingBlock(hidden_channels * 16 + hidden_channels * 8, num_classes, kernel_size=3, padding=1, activation='relu', SN=True)
        self.expand3 = ExpandingBlock(hidden_channels * 12 + hidden_channels * 4, num_classes, kernel_size=3, padding=1, activation='relu', SN=True)
        self.expand2 = ExpandingBlock(hidden_channels * 8 + hidden_channels * 2, num_classes, kernel_size=3, padding=1, activation='relu', SN=True)
        self.expand1 = ExpandingBlock(hidden_channels * 5 + hidden_channels, num_classes, kernel_size=3, padding=1, activation='relu', SN=True)
        self.downfeature = FeatureMapBlock(int(hidden_channels * 3), output_chan, stride=1, SN=True)

        self.sig = torch.nn.Sigmoid()

    def forward(self, noise_and_class, label):
        # convert the noise and class tensor of size (n_samples, input_dim) to a 4d tensor of size (n_samples, input_dim, 1, 1)
        x = self.fc(noise_and_class)
        x = x.view(-1, self.output_chan, self.img_height, self.img_width)  # Reshaping to (batch_size, 2, 13, 10)

        x0 = self.upfeature(x)
        x1 = self.contract1(x0, label)
        x2 = self.contract2(x1, label)
        x3 = self.contract3(x2, label)
        x4 = self.contract4(x3, label)
        y4 = self.expand4(x4, x3, label)  # Skip connection from contract1 to expand2
        y3 = self.expand3(y4, x2, label)  # Skip connection from contract1 to expand2
        y2 = self.expand2(y3, x1, label)  # Skip connection from upfeature to expand3
        y1 = self.expand1(y2, x0, label)  # Skip connection from upfeature to expand3
        y0 = self.downfeature(y1)
        yn = self.sig(y0)
        return yn


## model summary
# model = Generator_UNet(81, 13, 10, 2).to('cpu')  # move the model to GPU
# summary(model, input_size=(64, 81))


##
class Discriminator_Same(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_channels=32):
        super(Discriminator_Same, self).__init__()

        self.upfeature = FeatureMapBlock(input_channels, hidden_channels, SN=True)
        self.contract1 = ContractingBlock(hidden_channels, num_classes, use_norm=True, kernel_size=3, padding=1, activation='lrelu', SN=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, num_classes, kernel_size=3, padding=1, activation='lrelu', SN=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, num_classes, kernel_size=3, padding=1, activation='lrelu', SN=True)
        self.contract4 = ContractingBlock(hidden_channels * 8, num_classes, kernel_size=3, padding=1, activation='lrelu', SN=True)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, image_and_class, label):
        x = self.upfeature(image_and_class)
        x = self.contract1(x, label)
        x = self.contract2(x, label)
        x = self.contract3(x, label)
        x = self.contract4(x, label)
        x = self.final(x)
        return x

class Discriminator_Shrinking(nn.Module):
    def __init__(self, input_channels, num_classes, padding='valid', hidden_channels=32):
        super(Discriminator_Shrinking, self).__init__()

        self.upfeature = FeatureMapBlock(input_channels, hidden_channels, SN=True)
        self.contract1 = ContractingBlock(hidden_channels, num_classes, padding=padding, use_norm=True, activation='lrelu', SN=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, num_classes, padding=padding, activation='lrelu', SN=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, num_classes, padding=padding, activation='lrelu', SN=True)
        self.contract4 = ContractingBlock(hidden_channels * 8, num_classes, padding=padding, activation='lrelu', SN=True)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, image_and_class, label):
        x = self.upfeature(image_and_class)
        x = self.contract1(x, label)
        x = self.contract2(x, label)
        x = self.contract3(x, label)
        x = self.contract4(x, label)
        x = self.final(x)
        return x


## model summary
# model = Discriminator_Same(1).to('cpu')  # move the model to GPU
# summary(model, input_size=(64, 1, 13, 10))


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