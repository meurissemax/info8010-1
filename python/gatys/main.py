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

import os

# Neural networks with PyTorch
import torch

# Train or load pre-trained models
import torchvision.models as models

from images import img_load, img_save
from process import add_modules, run

from graphs import line_graph


##############
# Parameters #
##############

# Resources
style_path = 'resources/images/style/gaho.jpg'
content_path = 'resources/images/content/cow.jpg'
output_path = 'outputs/gaho-cow.png'

# Hyperparameters of the technique
model_name = 'vgg19'
num_steps = 200

weights = {
    'style': 1_000_000,
    'content': 10,
    'style_losses': [1] * 5,
    'content_losses': [1]
}

layers = {
    'content': ['conv_4'],
    'style': ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
}


########
# Main #
########

if __name__ == '__main__':
    ##################
    # Initialization #
    ##################

    # Print information
    print('Gatys et al. algorithm')
    print('----------------------\n')
    print('Initialization...')

    # Set the image size
    img_size = 512 if torch.cuda.is_available() else 128

    # Set the device (check if GPU are available, else use CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loads the model (only the features part, we don't need
    # the classifier) and put it in evaluation mode
    model = getattr(models, model_name)(pretrained=False).features.to(device).eval()

    # Set the normalization factor (to normalize the image
    # before sending it into the network)
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    ######################
    # Loading the images #
    ######################

    # Print information
    print('Loading images...')

    # Load the images and check if they have same dimensions
    img = {
        'style': img_load(style_path, img_size, device),
        'content': img_load(content_path, img_size, device)
    }

    # Create the input image (content image with white noise)
    img['input'] = torch.randn(img['content'].data.size(), device=device)

    #########################
    # Running the algorithm #
    #########################

    # Print information
    print('Running the algorithm (image size : {}, device : {})...'.format(img_size, device))

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
    output, style_scores, content_scores = run(style_model, img, num_steps, weights, losses)

    # Print information
    print('\nThe algorithm was executed successfully !')

    ###############
    # Exportation #
    ###############

    # Save the image
    img_save(output, output_path)

    # Plot and save a graph with losses evolution
    line_graph(
        list(range(1, len(style_scores) + 1)),
        [style_scores, content_scores],
        'Number of steps',
        'Loss',
        ['Style', 'Content'],
        os.path.splitext(output_path)[0] + '.pdf'
    )

    # Print general information
    print('Result saved as {}'.format(output_path))
