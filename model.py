import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models

import pdb

class MyNet(nn.Module):
    """Some Information about MyNet"""
    def __init__(self):
        super(MyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, padding = 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, padding = 1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3), 
            torch.nn.Dropout(p=0.25),

            nn.Conv2d(128, 128, 3, 2, padding = 1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, 2, padding = 1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3), 

            nn.Conv2d(256,512,3,2, bias=False),
            nn.BatchNorm2d(512),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*1*1,2)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x