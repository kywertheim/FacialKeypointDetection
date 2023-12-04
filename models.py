## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) #(224-5)/1+1=220
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4) #(110-4)/1+1=107
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3) #(53-3)/1+1=51
        self.conv4 = nn.Conv2d(128, 256, 2) #(25-2)/1+1=24
        self.conv5 = nn.Conv2d(256, 512, 2) #(12-2)/1+1=11
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512*5*5, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 136)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x=self.conv1(x)
        x=self.pool(F.relu(self.bn1(x)))
        x=self.conv2(x)
        x=self.pool(F.relu(self.bn2(x)))
        x=self.pool(F.relu(self.conv3(x)))
        x=self.pool(F.relu(self.conv4(x)))
        x=self.pool(F.relu(self.conv5(x)))
        x=x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))
        x=self.drop(x)
        x=F.relu(self.fc2(x))
        x=self.drop(x)
        x=self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
