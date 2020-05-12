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

# These classes implement the style and content losses
# required to perform neural style transfer. They are
# implemented from the original paper.

###########
# Imports #
###########

import torch
import torch.nn as nn
import torch.nn.functional as functional


#############
# Functions #
#############

def gram_matrix(tensor):
    """Computes and returns the Gram matrix obtained from tensor."""

    # Step 1 : vectorize the feature maps of the tensor

    # Size of the tensor (batch_size should be 1)
    batch_size, number_maps, h, w = tensor.size()

    # Vectorizing the tensor along w and h
    vectors = tensor.view(batch_size * number_maps, h * w)

    # Step 2 : computing the matrix via inner products

    return torch.mm(vectors, vectors.t())


###########
# Classes #
###########

class ContentLoss(nn.Module):
    def __init__(self, reference):
        """
        Initializes the content loss with the reference tensor.
        Since reference is not a variable in the computation tree,
        we need to detach it in order to avoid it being taken into
        account in the backpropagation.
        """

        super(ContentLoss, self).__init__()

        self.reference = reference.detach()

    def forward(self, x):
        """
        Computes the MSE between the reference and the tensor x and
        returns the input x so as not to interfere with the forward
        pass of the model.
        """

        self.loss = functional.mse_loss(x, self.reference)

        return x


class StyleLoss(nn.Module):
    def __init__(self, reference):
        """
        Initializes the style loss with the reference tensor.
        As for the content loss, the reference is detached before
        being assigned to a field.
        """

        super(StyleLoss, self).__init__()

        self.gram_reference = gram_matrix(reference).detach()

    def forward(self, x):
        """
        Computes the loss and returns the input tensor x.
        """

        a, b, c, d = x.size()

        self.loss = functional.mse_loss(
            input=gram_matrix(x),
            target=self.gram_reference
        )
        self.loss /= (2 * b ** 2 * (c * d) ** 2)

        return x
