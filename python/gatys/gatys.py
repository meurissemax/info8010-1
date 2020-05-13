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

# Neural networks with PyTorch
import torch

# Train or load pre-trained models
import torchvision.models as models

from gatys.images import img_save, img_load, img_optimizer
from gatys.process import add_modules, run


###########
# Classes #
###########

class Gatys:
    ###############
    # Constructor #
    ###############

    def __init__(self):
        # Print general information
        print('Gatys et al. algorithm')
        print('----------------------')
        print()

        ###################
        # Hyperparameters #
        ###################

        self.model_name = 'vgg19'

        # Desired size of the ouput image (depending on GPU availability)
        self.img_size = 512 if torch.cuda.is_available() else 128

        # Number of steps
        self.num_steps = 200

        # Weights
        self.weights = {
            'style': 1_000_000,
            'content': 10,
            'style_losses': [1] * 5,
            'content_losses': [1]
        }

        # Layers over which the losses are computed
        self.layers = {
            'content': ['conv_4'],
            'style': ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        }

    ####################
    # Public functions #
    ####################

    def initialize(self):
        # Print general information
        print('Initialization...')

        # Set the device (check if GPU are available, else use CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Loads the model (only the features part, we don't need
        # the classifier) and put it in evaluation mode
        self.model = getattr(models, self.model_name)(pretrained=False).features.to(self.device).eval()

        # Set the normalization factor (to normalize the image before sending it into the network)
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

    def run(self, style_path, content_path):
        # Print general information
        print('Loading images...')

        # Load the images and check if they have same dimensions
        self.img = {
            'style': img_load(style_path, self.img_size, self.device),
            'content': img_load(content_path, self.img_size, self.device)
        }

        # Create the input image (content image with white noise)
        self.img['input'] = torch.randn(self.img['content'].data.size(), device=self.device)

        # Add our loss and normalization modules in the model
        self.style_model, self.losses = add_modules(
            self.model,
            self.norm_mean,
            self.norm_std,
            self.img,
            self.layers,
            self.device
        )

        # Print general information
        print('Running the algorithm (device : {})...'.format(self.device))

        # Run the algorithm
        self.output, self.style_scores, self.content_scores = run(self.style_model, self.img, self.num_steps, self.weights, self.losses)

        # Print general information
        print()
        print('The algorithm was executed successfully !')

    def export(self, output_path, graph):
        # Save the image
        img_save(self.output, output_path)

        # Plot and save a graph with losses evolution
        graph(
            list(range(1, len(self.style_scores) + 1)),
            [self.style_scores, self.content_scores],
            'Number of steps',
            'Loss',
            ['Style', 'Content'],
            os.path.splitext(output_path)[0] + '.pdf'
        )

        # Print general information
        print('Result saved as {}'.format(output_path))
