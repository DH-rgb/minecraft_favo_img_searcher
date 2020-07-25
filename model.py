import torch
import torch.nn as nn 
import torch.nn.functional as F 

class MyModel(nn.Module):
    """Some Information about MyModel"""
    def __init__(self):
        super(MyModel, self).__init__()
        

    def forward(self, x):

        return x