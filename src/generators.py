import torch
import torch.nn as nn
import math

class FCGenerator(nn.Module):
    def __init__(self, image_size, channels, z_size):
        """
        Network which takes a batch of random vectors and creates images out of them.

        :param img_size: width and height of the image
        :param channels: number of channels
        """
        super(FCGenerator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.fc1 = nn.Linear(z_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, image_size*image_size*channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_batch):
        x = self.fc1(z_batch)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        image = x.reshape(-1, self.channels, self.image_size, self.image_size)
        return image

# Generator Code

class DCGANGenerator(nn.Module):
    def __init__(self, image_size, channels, z_size, ngf=64):
        super(DCGANGenerator, self).__init__()
        assert image_size & (image_size - 1) == 0
        self.z_size = z_size
        self.image_size = image_size
        self.channels = channels
        self.ngf = ngf
        self.n_blocks = int(math.log2(self.image_size)) - 1
        self.out_channel_multiplier = self.n_blocks - 2

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(
            nn.ConvTranspose2d(self.z_size, self.ngf * (2 ** self.out_channel_multiplier), 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * (2 ** self.out_channel_multiplier)),
            nn.ReLU(True)
        ))
        for i in range(1, self.n_blocks - 1):
            self.blocks.append(self.create_conv_block(self.ngf * (2 ** (self.out_channel_multiplier - i + 1)), self.ngf * (2 ** (self.out_channel_multiplier - i))))
        self.blocks.append(nn.Sequential(
            nn.ConvTranspose2d(self.ngf, self.channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
        ))
        
        # self.layers = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(self.z_size, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.ReLU(True),
        #     # state size. ``(ngf*8) x 4 x 4``
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),  
        #     # state size. ``(ngf*4) x 8 x 8``
        #     nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     # state size. ``(ngf*2) x 16 x 16``
        #     nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. ``(ngf) x 32 x 32``
        #     nn.ConvTranspose2d( ngf, self.channels, 4, 2, 1, bias=False),
        #     nn.Sigmoid()
        #     # state size. ``(nc) x 64 x 64``
        # )

    def create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, input):
        x = input.unsqueeze(-1).unsqueeze(-1)
        for block in self.blocks:
            x = block(x)
        return x
