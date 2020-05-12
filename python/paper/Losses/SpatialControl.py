# Implements the spatial control talked about on ...

import torch

# Computes the correct Fs as described in the paper and 
# returns them as an array of size R
def guided(input, guidanceChannels):
    # The Fr as described in the paper
    fr = []
    for channel in guidanceChannels:
        f = torch.Tensor(input.size())
        for column in range(input.size()[3]):
            f[:,column] = torch.mul(channel, input[:,column])
        fr.append(f)
    
    return fr


# Guided Gram matrices. Here, the input has already been vectorized
def gramMatrix(input):
    return torch.mm(input, input.t())



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

    def __init__(self, reference, guidanceChannels):
        """
        Initializes the style loss with the reference tensor.
        As for the content loss, the reference is detached before
        being assigned to a field.
        """
        super(StyleLoss, self).__init__()
        ref = guided(reference, guidanceChannels)
        self.gramReference = gramMatrix(ref).detach()

    def forward(self, x):
        """
        Computes the loss and returns the input tensor x.
        """
        a,b,c,d = x.size()
        # Change here too
        self.loss = functional.mse_loss(input=gramMatrix(x), target=self.gramReference)
        self.loss /= (2 * b**2 * (c*d)**2)
        
        return x