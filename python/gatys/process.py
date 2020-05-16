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

import copy

# Neural networks with PyTorch
import torch
import torch.nn as nn

# Efficient gradient descents
import torch.optim as optim

# To update the loss ration during process
from torch.optim.lr_scheduler import StepLR

# Losses and image optimizer
from losses import ContentLoss, StyleLoss


#############
# Functions #
#############

def add_modules(mod, mean, std, img, layers, device):
    """
    Modifiy the model to integrate new modules.
    """

    cnn_copy = copy.deepcopy(mod)
    norm_module = Normalization(mean, std).to(device)
    model = nn.Sequential(norm_module)

    content_losses = []
    style_losses = []

    i = 0

    for layer in cnn_copy.children():
        if isinstance(layer, nn.Conv2d):
            i += 1

            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            layer = nn.ReLU(inplace=False)

            name = 'relu_{}'.format(i)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError(
                'Unrecognized layer : {}'.format(
                    layer.__class__.__name__
                )
            )

        # Add the layer to our model
        model.add_module(name, layer)

        # Add the content layers at the right place
        if name in layers['content']:
            reference = model(img['content']).detach()
            content_loss = ContentLoss(reference)
            model.add_module('content_loss_{}'.format(i), content_loss)
            content_losses.append(content_loss)

        # Add the style layers at the right place
        if name in layers['style']:
            reference = model(img['style']).detach()
            style_loss = StyleLoss(reference)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)

    # Remove the layers after the last content and style ones
    for i in range(len(model) - 1, -1, -1):
        m = model[i]

        if isinstance(m, ContentLoss) or isinstance(m, StyleLoss):
            break

    model = model[:(i + 1)]

    return model, {'style': style_losses, 'content': content_losses}


def run(model, img, num_steps, weights, losses):
    """
    Run the Gatys et al. algorithm.
    """

    # Adds the input image to the gradient descent
    optimizer = optim.LBFGS([img['input'].requires_grad_()])

    # Set a decaying learning rate
    scheduler = StepLR(optimizer, step_size=50, gamma=0.3)

    # Save the scores
    style_scores = []
    content_scores = []

    run = [0]

    while run[0] < num_steps:
        def closure():
            # Steps in the scheduler
            scheduler.step()

            # Limits the values of the updates image
            img['input'].data.clamp_(0, 1)

            # Reset the gradients to zero before the backpropagation
            optimizer.zero_grad()
            model(img['input'])

            style_score = 0
            content_score = 0

            for loss, weight in zip(losses['style'], weights['style_losses']):
                style_score += loss.loss * weight

            for loss, weight in zip(losses['content'], weights['content_losses']):
                content_score += loss.loss * weight

            style_score *= weights['style']
            content_score *= weights['content']

            style_scores.append(style_score)
            content_scores.append(content_score)

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            step = int((run[0] / (num_steps)) * 50)
            print('[Progress : {}/{}] [{}{}]'.format(
                str(run[0]).rjust(len(str((num_steps)))),
                (num_steps),
                '=' * step, ' ' * (50 - step)
            ), end='\r')

            return content_score + style_score

        optimizer.step(closure)

    img['input'].data.clamp_(0, 1)

    return img['input'], style_scores, content_scores


###########
# Classes #
###########

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        # Update the sizes
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std
