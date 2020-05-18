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
import glob

import torch

import torchvision.models as models

from images import img_load, img_save
from process import add_modules, run

from graphs import line_graph


#############
# Resources #
#############

# Path to resources (can be a path to a single
# image or a folder)
style_path = 'resources/images/style/'
content_path = 'resources/images/content/'

# Path to export outputs (must be a folder)
output_path = 'outputs/'


###################
# Hyperparameters #
###################

# Model
model_name = 'vgg19'
model_pretrained = True

# Number of steps
num_steps = 300

# Weights
weights = {
    'style': 1_000_000,
    'content': 10,
    'style_losses': [1, 0.8, 0.6, 0.4, 0.2],
    'content_losses': [1]
}

# Layers
layers = {
    'style': ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13'],
    'content': ['conv_10']
}

# Flag to control the replacement of the
# MaxPool2d layers by AvgPool2d layers
replace_max_to_avg = True

# Scheduler
scheduler = {
    'step_size': 50,
    'gamma': 0.3
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
    print('Image size : {}'.format(img_size))

    # Set the device (check if GPU are available, else use CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device : {}'.format(device))

    # Loads the model (only the features part, we don't need
    # the classifier) and put it in evaluation mode
    model = getattr(models, model_name)(pretrained=model_pretrained).features.to(device).eval()
    print('Model name : {} ({})'.format(model_name, 'pretrained' if model_pretrained else 'not pretrained'))

    # Set the normalization factor (to normalize the image
    # before sending it into the network)
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    print()

    ######################
    # Loading the images #
    ######################

    # Print information
    print('Searching for images...')

    # Get style image(s)
    style_paths = []

    if os.path.isfile(style_path):
        style_paths += [style_path]
    else:
        style_paths = glob.glob('{}*.png'.format(style_path))

    print('{} style image(s) found'.format(len(style_paths)))

    # Get content images
    content_paths = []

    if os.path.isfile(content_path):
        content_paths += [content_path]
    else:
        content_paths = glob.glob('{}*.png'.format(content_path))

    print('{} content image(s) found'.format(len(content_paths)))

    print()

    #########################
    # Running the algorithm #
    #########################

    for style in style_paths:
        for content in content_paths:

            # Print information
            print('Style image : {}'.format(style))
            print('Content image : {}'.format(content))
            print('Loading images...')

            # Load the images and check if they have same dimensions
            img = {
                'style': img_load(style, img_size, device),
                'content': img_load(content, img_size, device)
            }

            # Create the input image
            img['input'] = img['content'].clone()

            # Print information
            print('Running the algorithm...')

            # Add our loss and normalization modules in the model
            style_model, losses = add_modules(model, norm_mean, norm_std, img, layers, device, replace_max_to_avg)

            # Run the algorithm
            output, style_scores, content_scores = run(style_model, img, num_steps, weights, losses, scheduler)

            # Print information
            print('\nThe algorithm was executed successfully !')

            ###############
            # Exportation #
            ###############

            # Get the full output path
            full_output_path = '{}{}-{}-{}-{}-{}-{}'.format(
                output_path,
                os.path.splitext(os.path.basename(style))[0],
                os.path.splitext(os.path.basename(content))[0],
                model_name,
                'pretrained' if model_pretrained else 'notpretrained',
                'avg' if replace_max_to_avg else 'max',
                num_steps
            )

            # Save the image
            img_save(output, full_output_path + '.png')

            # Save losses values
            with open(full_output_path + '-style-loss.txt', 'w') as f:
                f.write('\n'.join(list(map(str, style_scores))))

            with open(full_output_path + '-content-loss.txt', 'w') as f:
                f.write('\n'.join(list(map(str, content_scores))))

            # Plot and save a graph with losses evolution
            line_graph(
                list(range(1, len(style_scores) + 1)),
                [style_scores, content_scores],
                'Number of steps',
                'Loss',
                ['Style', 'Content'],
                full_output_path + '.pdf'
            )

            # Print general information
            print('Result saved as {}[.png|.pdf|.txt]\n'.format(full_output_path))
