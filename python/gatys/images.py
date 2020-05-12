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

###########
# Imports #
###########

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image


#############
# Functions #
#############

def img_save(tensor, path):
    """Save an image."""

    img = tensor.cpu()
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)

    img.save(path)


def img_load(name, size, device):
    """Load an image."""

    ld = transforms.Compose([
        transforms.Resize([size, size]),
        transforms.ToTensor()
    ])

    img = Image.open(name)
    img = ld(img).unsqueeze(0)
    img = img.to(device, torch.float)

    # If there are 4 channels (for example alpha channel
    # of PNG images), we discard it
    if img.size()[1] > 3:
        img = img[:, :3, :, :]

    return img


def img_optimizer(img):
    """Provide an optimizer for the gradient descent."""

    # Read https://pytorch.org/docs/stable/notes/autograd.html
    # Must be part of the gradient descent since we are creating
    # this image iteratively

    return optim.LBFGS([img.requires_grad_()])
