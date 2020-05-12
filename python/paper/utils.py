import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from PIL import Image
import torchvision.transforms as transforms
import copy
from Losses.InitialLosses import ContentLoss, StyleLoss

# Image saver
def imSave(tensor, filename):
    image = tensor.cpu()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save("outputs/"+filename+".png")


# Image loader
def loader(imgName, imSize, device):
    ld = transforms.Compose([
        transforms.Resize([imSize, imSize]),
        transforms.ToTensor()
    ])

    img = Image.open(imgName)
    img = ld(img).unsqueeze(0)
    img = img.to(device, torch.float)
    # If there are 4 channels (for example alpha channel of PNG images),
    # we discard it
    if img.size()[1] > 3:
        img = img[:, :3, :, :]
    return img


# Image normalizer
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Change the sizes
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)
    
    def forward(self, x):
        return (x-self.mean)/self.std


# Modifies the model to integrate our modules
def addModules(mod, mean, std, styleImg, contentImg, 
                contentLayers, styleLayers, device):

    cnnCopy = copy.deepcopy(mod)
    normModule = Normalization(mean, std).to(device)
    model = nn.Sequential(normModule)

    # Will contain the ...
    contentLosses = []
    styleLosses = []

    i = 0
    for layer in cnnCopy.children():
        if isinstance(layer, nn.Conv2d):
            i+=1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        # To change to add mean pooling for example
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer : {layer.__class__.__name__}')
        
        # Add the layer to our model
        model.add_module(name, layer)

        # Add the content layers at the right place
        if name in contentLayers:
            reference = model(contentImg).detach()
            contentLoss = ContentLoss(reference)
            model.add_module(f"content_loss_{i}", contentLoss)
            contentLosses.append(contentLoss)
        
        # Add the style layers at the right place
        if name in styleLayers:
            reference = model(styleImg).detach()
            styleLoss = StyleLoss(reference)
            model.add_module(f"style_loss_{i}", styleLoss)
            styleLosses.append(styleLoss)
    
    # Remove the layers after the last content and style ones
    for i in range(len(model)-1,-1,-1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i+1)]

    return model, styleLosses, contentLosses


# Provides an optimizer for the gradient descent
def imgOptimizer(img):
    # Read https://pytorch.org/docs/stable/notes/autograd.html
    # Must be part of the gradient descent since we are creating this image
    # iteratively
    return optim.LBFGS([img.requires_grad_()])


def runStyleTransfer(model, inputImg, contentImg, styleImg, numSteps, styleWeight, contentWeight, styleLosses, contentLosses, sLossesWeights, cLossesWeights):
    
    # Adds the input image to the gradient descent
    optimizer = imgOptimizer(inputImg)

    # Set a decaying learning rate
    scheduler = StepLR(optimizer, step_size=50, gamma=0.3)

    run = [0]
    while run[0] < numSteps:

        def closure():
            # Steps in the scheduler
            scheduler.step()

            # Limits the values of the updates image
            inputImg.data.clamp_(0,1)

            # Reset the gradients to zero before the backprop
            optimizer.zero_grad()
            model(inputImg)
            styleScore = contentScore = 0

            for id, loss in enumerate(styleLosses):
                styleScore += loss.loss * sLossesWeights[id]
            
            for id, loss in enumerate(contentLosses):
                contentScore += loss.loss * cLossesWeights[id]
            
            styleScore *= styleWeight
            contentScore *= contentWeight

            loss = styleScore + contentScore
            loss.backward()
            
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"run {run}")
                print(f"Style loss : {styleScore}; Content Loss : {contentScore}\n")
            
            return contentScore + styleScore
        
        optimizer.step(closure)
    
    inputImg.data.clamp_(0,1)

    return inputImg
