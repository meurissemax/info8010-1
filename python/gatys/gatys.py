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

from __future__ import print_function

import os

import torch
import torchvision.models as models

from gatys.images import img_save, img_load, img_optimizer
from gatys.process import add_modules, run


#####################
# General variables #
#####################

# Hyperparameters
img_size = 512 if torch.cuda.is_available() else 128
num_steps = 500

weights = {
    'style': 1_000_000,
    'content': 10,
    'style_losses': [1] * 5,
    'content_losses': [1]
}

# Layers over which the losses are computed
layers = {
    'content': ['conv_4'],
    'style': ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
}


#############
# Functions #
#############

def gatys(style_path, content_path, output_path, graph):
    # Print general information
    print('Gatys et al. algorithm')
    print('----------------------')
    print()
    print('Initialization...')

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loads the model (only the features part, we don't need
    # the classifier) and put it in evaluation mode
    model = models.vgg19(pretrained=False).features.to(device).eval()

    # Set the norm
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Load the images
    img = {
        'style': img_load(style_path, img_size, device),
        'content': img_load(content_path, img_size, device)
    }

    img['input'] = torch.randn(img['content'].data.size(), device=device)

    # Add our loss and normalization modules in the model
    style_model, losses = add_modules(
        model,
        norm_mean,
        norm_std,
        img,
        layers,
        device
    )

    # Run the algorithm
    print('Running the algorithm...')
    output, style_scores, content_scores = run(style_model, img, num_steps, weights, losses)
    print()
    print('The algorithm was executed successfully !')

    # Save the result
    img_save(output, output_path)
    graph(
        list(range(1, len(style_scores) + 1)),
        [style_scores, content_scores],
        'Number of steps',
        'Loss',
        ['Style', 'Content'],
        os.path.splitext(output_path)[0] + '.pdf'
    )
    print('Result saved as {}'.format(output_path))
