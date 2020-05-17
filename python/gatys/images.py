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
import torchvision.transforms as transforms

from PIL import Image


#############
# Functions #
#############

def img_load(path, size, device):
    """
    Load an image from path, resize it, transform it
    to tensor and send it to device.
    """

    # Create a transformation to transform a PIL image
    # (represented by values between 0 and 255) into a
    # tensor (values between 0 and 1)
    loader = transforms.Compose([
        transforms.Resize([size, size]),
        transforms.ToTensor()
    ])

    # Open the image
    img = Image.open(path)

    # Fake batch dimension required to fit network's input dimensions
    img = loader(img)[:3, :, :].unsqueeze(0)

    return img.to(device, torch.float)


def img_save(tensor, path):
    """
    Take as input a tensor representing an image,
    convert it to a PIL image and save it to path.
    """

    # We create the unloader
    unloader = transforms.ToPILImage()

    # We clone the tensor to not do changes on it
    img = tensor.cpu().clone()

    # Remove the fake batch dimension
    img = img.squeeze(0)

    # Transform the copied tensor to PIL image
    img = unloader(img)

    # Save the image
    img.save(path)
