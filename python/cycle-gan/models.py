"""
INFO8010-1 - Deep learning
University of Liège
Academic year 2019-2020

Project : neural style transfer

Authors :
    - Maxime Meurisse
    - Adrien Schoffeniels
    - Valentin Vermeylen
"""

import torch
import torch.nn as nn

# Knowledge to create residual blocks taken from 
# https://github.com/trailingend/pytorch-residual-block/blob/master/main.py
class Residual(nn.Module):

    def __init__(self):
        super(Residual, self).__init__()

        self.resBlock = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256,256,3),
            nn.InstanceNorm2d(256)
        )
    
    def forward(self, input_):
        return input_ + self.resBlock(input_)


class Generator(nn.Module):
    """Generator part of the network. Implemented from the appendix of
    https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, inputSize):
        # inputSize is a tuple containing the number of channels, 
        # the width and the height of the input image.

        super(Generator, self).__init__()
        nbChannels, width, height = inputSize

        layers = []

        # Add the first convolutional module
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(nbChannels, 64, 7, stride=1))
        # Instance instead of batch, as stated in CycleGan paper
        layers.append(nn.InstanceNorm2d(64))
        layers.append(nn.ReLU(True))

        # Add the two next convolutional layers
        layers.append(nn.Conv2d(64, 128, 3, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(128))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(128, 256, 3, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(256))
        layers.append(nn.ReLU(True))

        # Add the residual blocks
        nbBlocks = (6 if height == 128 else 9)
        for i in range(nbBlocks):
            layers.append(Residual())

        # Add the half-strided convolutions
        layers.append(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1))
        layers.append(nn.InstanceNorm2d(128))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1))
        layers.append(nn.InstanceNorm2d(64))
        layers.append(nn.ReLU(True))

        # Add the last convolutional layer
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(64, 3, 7, stride=1))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)
    
    def forward(self, input_):
        return self.model(input_)


class Discriminator(nn.Module):
    """Discriminator part of the network. # Implemented from https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self):

        super(Discriminator, self).__init__()

        layers = []

        # Add the first block
        layers.append(nn.Conv2d(3, 64, 4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        # Add the next 3 layers
        layers.append(nn.Conv2d(64, 128, 4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(128))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(128, 256, 4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(256))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(256, 512, 4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(512))
        layers.append(nn.LeakyReLU(0.2))

        # Add the next layer (the idea to include this one comes from https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
        # and has been confirmed by comparing our implementation with the one of the authors afterwards)
        layers.append(nn.Conv2d(512, 512, 4, stride=1, padding=1))
        layers.append(nn.InstanceNorm2d(512))
        layers.append(nn.LeakyReLU(0.2))

        # Add the last layer to get a 1D output
        layers.append(nn.Conv2d(512, 1, 4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, input_):
        return self.model(input_)
