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

# Use dataset.sh to first obtain the datasets used here.

import torch
import itertools
import argparse
import numpy as np
import os

from torch.utils.data import DataLoader

from models import Generator, Discriminator
from utils import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Add arguments to the CycleGAN script.")
    parser.add_argument(
        '--epochs',
        help="Number of epochs of the training",
        type=int,
        default=200,
        required=False)
    parser.add_argument(
        '--offset',
        help="Offset for the training (at which epoch do we start decreasing linearly)",
        type=int,
        default=100,
        required=False)
    parser.add_argument(
        '--dataset',
        help="Name of the dataset",
        type=str,
        default="/monet2photo",
        required=False)
    parser.add_argument(
        '--Path2Dataset',
        help="Path to the dataset folder",
        type=str,
        default="../datasets",
        required=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imSize = 256 if torch.cuda.is_available() else 128

    # Define the two generators and discriminators

    # GXY : Takes images from X and translates them to Y domain
    GAB = Generator((3, imSize, imSize)).double().to(device)
    GBA = Generator((3, imSize, imSize)).double().to(device)

    DA = Discriminator().double().to(device) # Discriminate A images
    DB = Discriminator().double().to(device) # Discriminate B images

    # Initialize the weights of the networks as described in the paper
    GAB.apply(init_weights)
    GBA.apply(init_weights)
    DA.apply(init_weights)
    DB.apply(init_weights)

    # Select the different losses 
    LGan = torch.nn.MSELoss().to(device)
    LCyc = torch.nn.L1Loss().to(device)
    LId = torch.nn.L1Loss().to(device)

    # Create the optimizers
    # We have to chain because the losses make use of both networks
    optimGen = torch.optim.Adam(itertools.chain(GAB.parameters(), GBA.parameters()), lr = 0.0002, betas=(0.5,0.999))
    optimDis = torch.optim.Adam(itertools.chain(DA.parameters(), DB.parameters()), lr=0.0001, betas=(0.5,0.999))

    # Create custom lr schedulers since they have a particular behaviour
    schedGen = torch.optim.lr_scheduler.LambdaLR(optimGen, lr_lambda = CustomLR(args.epochs, args.offset).step)
    schedDis = torch.optim.lr_scheduler.LambdaLR(optimDis, lr_lambda = CustomLR(args.epochs, args.offset).step)

    # Create the datasets (although only one will be used for the training)
    datasetTrain = DataLoader(load_datasets(imSize, args.Path2Dataset+args.dataset+'/train'), batch_size=3, shuffle=True)
    datasetTest = DataLoader(load_datasets(imSize, args.Path2Dataset+args.dataset+'/test'), batch_size=3, shuffle=True)
    
    # Create the pools of previously created images for discriminator training
    poolA = []
    poolB = []

    # Sizes for the ground truths
    GBAOutputSize = None
    DOutputSize = None

    # Start the training
    for epoch in range(args.epochs):
        for index, batch in enumerate(datasetTrain):

            # Load true, real images
            trueA = batch["A"].double().to(device).detach()
            trueB = batch["B"].double().to(device).detach()
            
            GBAOutputSize = GBAOutputSize if GBAOutputSize is not None else GBA(trueA).size()
            DOutputSize = DOutputSize if DOutputSize is not None else DA(GAB(trueA)).size()

            # Create the ground truth of true and false images
            trDOut = torch.tensor(np.ones(DOutputSize), requires_grad=False).to(device)
            faDOut = torch.tensor(np.zeros(DOutputSize), requires_grad=False).to(device)
            
            ## Train the generators

            # Fake generated images
            fakeA = GBA(trueB)
            fakeB = GAB(trueA)
            
            # Cycle losses (going back and forth to a domain)
            cLoss1 = LCyc(GBA(GAB(trueA)), trueA)
            cLoss2 = LCyc(GAB(GBA(trueB)), trueB)
            cLoss = (cLoss1 + cLoss2)/2

            # Adversarial losses
            gLoss1 = LGan(DA(fakeA), trDOut)
            gLoss2 = LGan(DB(fakeB), trDOut)
            lG = (gLoss1 + gLoss2)/2

            # Identity losses
            idLoss1 = LId(GAB(trueB), trueB)
            idLoss2 = LId(GBA(trueA), trueA)
            idL = (idLoss1 + idLoss2)/2

            lossG = lG + cLoss + idL
            optimGen.zero_grad()
            lossG.backward()
            optimGen.step()
            
            ## Train the discriminators

            # Detach the generated images
            fakeA = fakeA.detach()
            fakeB = fakeB.detach()

            ## A

            # Update the pools
            fakeA = fakeA.to("cpu") # For memory on the cluster
            fakeB = fakeB.to("cpu")
            fakeA = update_pool(poolA, fakeA).to(device)
            fakeB = update_pool(poolB, fakeB).to(device)

            # Compute the losses
            realLoss = LGan(DA(trueA), trDOut)
            fakeLoss = LGan(DA(fakeA), faDOut)
            lossDA = (realLoss + fakeLoss)/2

            ## B

            # Compute the losses
            realLoss = LGan(DB(trueB), trDOut)
            fakeLoss = LGan(DB(fakeB), faDOut)
            lossDB = (realLoss + fakeLoss)/2

            lossD = lossDA + lossDB
            optimDis.zero_grad()
            lossD.backward()
            optimDis.step()

            #print("Epoch : ", epoch, " Batch : ", index, " lossG : ", lossG, " LossD : ", lossD)

        ## Update the learning rates
        schedGen.step()
        schedDis.step()
        
        if epoch % 20 == 0 or epoch == (args.epochs-1):
            # Save model weights
            torch.save(GAB.state_dict(), "/home/mmeurisse/Valentin/CGAN/saved_models%s/GAB_%d.pth" % (args.dataset, epoch))
            torch.save(GBA.state_dict(), "/home/mmeurisse/Valentin/CGAN/saved_models%s/GBA_%d.pth" % (args.dataset, epoch))
            torch.save(DA.state_dict(), "/home/mmeurisse/Valentin/CGAN/saved_models%s/DA_%d.pth" % (args.dataset, epoch))
            torch.save(DB.state_dict(), "/home/mmeurisse/Valentin/CGAN/saved_models%s/DB_%d.pth" % (args.dataset, epoch))
            if epoch != 0 and epoch != (args.epochs-1):
                os.remove("/home/mmeurisse/Valentin/CGAN/saved_models%s/GAB_%d.pth" % (args.dataset, epoch-20))
                os.remove("/home/mmeurisse/Valentin/CGAN/saved_models%s/GBA_%d.pth" % (args.dataset, epoch-20))
                os.remove("/home/mmeurisse/Valentin/CGAN/saved_models%s/DA_%d.pth" % (args.dataset, epoch-20))
                os.remove("/home/mmeurisse/Valentin/CGAN/saved_models%s/DB_%d.pth" % (args.dataset, epoch-20))
