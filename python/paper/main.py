from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

from Losses.InitialLosses import ContentLoss, StyleLoss
from utils import loader, addModules, imgOptimizer, runStyleTransfer, imSave

# Launch the neural style transfer script
if __name__ == "__main__":

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Hyperparameters
    imSize = 512 if torch.cuda.is_available() else 128
    numSteps = 500
    styleWeight = 1000000
    contentWeight = 10
    sLossesWeights = [1] * 5
    cLossesWeights = [1]

    
    normMean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normStd = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    # Layers over which the losses are computed
    contentLayers = ['conv_4']
    styleLayers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # Load the images
    st_name = "starry-night"
    ct_name = "eiffel-tower"
    styleImg = loader("../../resources/style/"+st_name+".png", imSize, device)
    contentImg = loader("../../resources/content/"+ct_name+".png", imSize, device)
    inputImg = torch.randn(contentImg.data.size(), device=device)

    # Loads the model (only the features part, we don't need the classifier)
    # And put it in evaluation mode (!= training mode)
    modelName = "vgg19"
    model = models.vgg19(pretrained=False).features.to(device).eval()

    # Add our loss and normalization modules in the model
    styleModel, stLosses, ctLosses = addModules(model, normMean, normStd, styleImg, contentImg, contentLayers, styleLayers, device)

    # Run the algorithm
    output = runStyleTransfer(styleModel, inputImg, contentImg, styleImg, numSteps, styleWeight, contentWeight, stLosses, ctLosses, sLossesWeights, cLossesWeights)
    imSave(output, st_name+ct_name+modelName)
