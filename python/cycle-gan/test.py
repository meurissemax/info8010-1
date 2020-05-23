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

# This file simply loads a trained generator and saves an 
# image passing through it.

# We provide one generator model, the one that currently gives the best results
# GBA_99_oneModelId is the model for only one generator, the identity loss and the plateau LR scheduler

import torch
from PIL import Image
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from models import Generator, Discriminator
from utils import *

# Image saver
def imSave(tensor, filename):

    image = tensor.cpu().detach()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(filename+".png")


if __name__ == "__main__":

    imSize = 128
    generator = "GBA_160_notd.pth"
    imgName = "test.jpg"
    outName = "notd"

    # Load the generator
    GBA = Generator((3, imSize, imSize)).double()
    GBA.load_state_dict(torch.load(generator, map_location=torch.device('cpu')))

    transform = transforms.Compose([
        transforms.Resize([imSize, imSize]),
        transforms.ToTensor()
    ])

    img = transform(Image.open(imgName))

    imgOut = GBA(img.unsqueeze(0).double())

    imSave(imgOut.float(), outName)
