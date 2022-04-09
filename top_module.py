#Import all the header files
import redcaps

import sys

sys.path.insert(1, "/Users/vineet/Desktop/Winter -22 Courses/EECS 545/Project/545-ML-Project/redcaps")

from model import *

#Import the pytorch 
import torch 


#Import numpy as it is required by the GRADCAM to generate visulizations and suerimpose them 
import numpy as np 

#import modules required by GradCAM 
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

########################################### MAIN CODE STARTS FROM HERE #############################################################

#Create Model Objects, Image Loaders 

virtexModel, imageLoader, sample_images, valid_subs=create_objects()

# Define output class Wrapper for the Virtex model that is required by the GradCAM 
class VirtexModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(VirtexModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        tmp=self.model.predict(x) 
        print(tmp[1])
        #return self.model.predict(x)[2].reshape(1,-1)
        return tmp[2].reshape(1,-1)
    
model = VirtexModelOutputWrapper(virtexModel) 


#Load the input image (Just one image for now)
#TODO: Load the entire dataset and pass it to the function to generate Visulizations for all
