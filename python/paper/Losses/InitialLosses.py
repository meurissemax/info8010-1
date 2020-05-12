"""
These classes implement the style and content losses
required to perform neural style transfer. They are
implemented from the original paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional

# Computes and returns the Gram matrix obtained from tensor.
def gramMatrix(tensor):
    ## Step 1 : Vectorize the feature maps of the tensor

    # Size of the tensor (batchSize should be 1)
    batchSize, nbMaps, h, w = tensor.size()
    # Vectorizing the tensor along w and h
    vectors = tensor.view(batchSize*nbMaps, h*w)

    ## Step 2 : Computing the matrix via inner products

    gram = torch.mm(vectors, vectors.t())

    return gram


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
        self.gramReference = gramMatrix(reference).detach()

    def forward(self, x):
        """
        Computes the loss and returns the input tensor x.
        """
        a,b,c,d = x.size()

        self.loss = functional.mse_loss(input=gramMatrix(x), target=self.gramReference)
        self.loss /= (2 * b**2 * (c*d)**2)
        
        return x