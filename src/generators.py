import os
import torch
import torch.nn as nn
import math

from config import TrainingConfig

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

    @classmethod
    def from_config(cls,config:TrainingConfig):
        return cls(image_size=config.image_size,channels=config.channels,z_size=config.z_size)

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

    @classmethod
    def from_config(cls,config:TrainingConfig):
        return cls(image_size=config.image_size,channels=config.channels,z_size=config.z_size,ngf=config.ngf)

    @classmethod
    def from_ckpt(cls,ckpt_dir:str):
        config_path=os.path.join(ckpt_dir,"config.yaml")
        config=TrainingConfig.from_yaml(config_path)
        generator=DCGANGenerator.from_config(config)
        weight_path = os.path.join(ckpt_dir, "x_ray_generator.pt")
        generator.load_state_dict(torch.load(weight_path,map_location="cpu"))
        return generator
    
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
