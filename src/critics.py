import torch
import torch.nn as nn
import math

from config import TrainingConfig

class FCCritic(nn.Module):
    def __init__(self, image_size, channels):
        """
        Neural network which takes a batch of images and creates a batch of scalars which represent a score for how
        real the image looks.
        Uses just several fully connected layers.
        Works for arbitrary image size and number of channels, because it flattens them first.

        :param img_size:
        :param channels: number of channels in the image (RGB = 3, Black/White = 1)
        """
        super(FCCritic, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.fc1 = nn.Linear(image_size*image_size*channels, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    @classmethod
    def from_config(cls,config:TrainingConfig):
        return cls(image_size=config.image_size,channels=config.channels)

    def forward(self, image_batch):
        """
        Method which performs the computation.

        :param image: Tensor of shape [batch_size, self.img_size, self.img_size, self.channels]
        :return: Tensor of shape [batch_size, 1]
        """
        x = image_batch.reshape(-1, self.img_size*self.img_size*self.channels)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.fc3(x)
    
class DCGANCritic(nn.Module):
    def __init__(self, image_size, channels, ngf=64):
        super(DCGANCritic, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.ngf = ngf
        self.n_blocks = int(math.log2(self.image_size)) - 1

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(
            nn.Conv2d(self.channels, self.ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        for i in range(1, self.n_blocks - 1):
            self.blocks.append(self.create_conv_block(self.ngf * (2 ** (i - 1)), self.ngf * (2 ** i)))
        self.blocks.append(nn.Sequential(
            nn.Conv2d(self.ngf * (2 ** (self.n_blocks - 2)), 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ))
    
    @classmethod
    def from_config(cls,config:TrainingConfig):
        return cls(image_size=config.image_size,channels=config.channels,ngf=config.ngf)


    def create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        x = input
        for block in self.blocks:
            x = block(x)
        return torch.squeeze(x)


class ConvCritic(nn.Module):
    def __init__(self, img_size, channels):
        raise
