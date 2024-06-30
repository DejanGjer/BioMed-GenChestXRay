import torch
import torch.nn as nn
import math

class FCCritic(nn.Module):
    def __init__(self, img_size, channels):
        """
        Neural network which takes a batch of images and creates a batch of scalars which represent a score for how
        real the image looks.
        Uses just several fully connected layers.
        Works for arbitrary image size and number of channels, because it flattens them first.

        :param img_size:
        :param channels: number of channels in the image (RGB = 3, Black/White = 1)
        """
        super(FCCritic, self).__init__()
        self.img_size = img_size
        self.channels = channels

        self.fc1 = nn.Linear(img_size*img_size*channels, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

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

# class DCGANCritic(nn.Module):
#     def __init__(self, img_size, channels):
#         """
#         DCGAN is only defined for 64x64 images, it takes the img_size and channels here only not to break the interface

#         :param img_size:
#         :param channels:
#         """
#         super(DCGANCritic, self).__init__()
#         assert img_size == 64, "Works only for 64x64 images"
#         self.img_size = img_size
#         self.channels = channels

#         kernel_size = (5,5)
#         stride = (2,2)
#         padding_mode = "replicate"
#         self.conv1 = nn.Conv2d(self.channels, 64, kernel_size, stride, padding_mode=padding_mode)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size, stride, padding_mode=padding_mode)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size, stride, padding_mode=padding_mode)
#         self.conv4 = nn.Conv2d(256, 1024, kernel_size, stride, padding_mode=padding_mode)
#         self.fc1 = nn.Linear(4*4*1024, 1)
#         self.relu = nn.ReLU()

#     def forward(self, image):
#         """
#         Works only for 64x64

#         :param image:
#         :param reuse:
#         :return:
#         """
#         x = self.conv1(image)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.conv4(x)
#         x = self.relu(x)

#         x = self.fc1(x.flatten())
#         return x
    
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


        # self.main = nn.Sequential(
        #     # input is ``(nc) x 64 x 64``
        #     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. ``(ndf) x 32 x 32``
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. ``(ndf*2) x 16 x 16``
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. ``(ndf*4) x 8 x 8``
        #     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. ``(ndf*8) x 4 x 4``
        #     nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
        # )

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
