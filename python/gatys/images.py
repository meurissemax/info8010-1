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

# Neural networks with PyTorch
import torch

# Efficient gradient descents
import torch.optim as optim

# Transform PIL images into tensors
import torchvision.transforms as transforms

# Load and display images
from PIL import Image


#############
# Functions #
#############

def img_save(tensor, path):
    """Save an image."""

    # We clone the tensor to not do changes on it
    img = tensor.cpu().clone()

    # Remove the fake batch dimension
    img = img.squeeze(0)

    # Transform the copied tensor to PIL image
    img = transforms.ToPILImage()(img)

    # Save the image
    img.save(path)


def img_load(name, size, device):
    """Load an image."""

    # Create a transformation to transform a PIL image
    # (represented by values between 0 and 255) into a
    # tensor (values between 0 and 1)
    loader = transforms.Compose([
        transforms.Resize([size, size]),
        transforms.ToTensor()
    ])

    img = Image.open(name)

    # Fake batch dimension required to fit network's input dimensions
    img = loader(img).unsqueeze(0)

    # If there are 4 channels (for example alpha channel
    # of PNG images), we discard it
    if img.size()[1] > 3:
        img = img[:, :3, :, :]

    return img.to(device, torch.float)


def img_optimizer(img):
    """Provide an optimizer for the gradient descent."""

    # Read https://pytorch.org/docs/stable/notes/autograd.html
    # Must be part of the gradient descent since we are creating
    # this image iteratively

    return optim.LBFGS([img.requires_grad_()])