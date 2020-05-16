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
import torch.nn as nn
import torch.nn.functional as functional


#############
# Functions #
#############

def gram_matrix(tensor):
    """Computes and returns the Gram matrix obtained from tensor."""

    # Vectorize the feature maps of the tensor
    batch_size, number_maps, h, w = tensor.size()

    # Vectorizing the tensor along w and h
    vectors = tensor.view(batch_size * number_maps, h * w)

    # Computing the matrix via inner products
    gram = torch.mm(vectors, vectors.t())

    # Normalize values of the Gram matrix by dividing by the
    # number of elements in each feature maps
    return gram.div(batch_size * number_maps * h * w)


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

        # Compute the Gram matrix
        gram = gram_matrix(x)

        # Compute the loss
        self.loss = functional.mse_loss(gram, self.gram_reference)

        return x
