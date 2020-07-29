import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms

import os 
from glob import glob 
from PIL import Image

class ResNet_Dataset(torch.utils.data.Dataset):
    def __init__(self, datadir, is_train, image_size=224):
        super(torch.utils.data.Dataset, self).__init__()
        if is_train:
            self.image_path = [p for p in glob(datadir+"train/**/*.jpg")]
        else:
            self.image_path = [p for p in glob(datadir+"test/**/*.jpg")]
        self.len = len(self.image_path)
        self.image_size = image_size
        if is_train:
            self.transformer = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transformer = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        p = self.image_path[index]

        image = Image.open(p).convert('RGB')
        image = self.transformer(image)

        label = p.split("/")[2]
        label = 1 if label=="favo" else 0
        
        return image, label


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, is_train, image_size=512):
        super(torch.utils.data.Dataset,self).__init__()
        self.image_size = image_size
        if is_train:
            self.image_path = [p for p in glob(datadir+"train/**/*.jpg")]
        else:
            self.image_path = [p for p in glob(datadir+"test/**/*.jpg")]
        self.transformers = self.make_transformers(is_train)
        self.len = len(self.image_path)
        

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        p = self.image_path[index]

        image = Image.open(p).convert('RGB')
        image_z = self.transformers(image)

        label = p.split("/")[2]
        label = 1 if label=="favo" else 0
        
        return image_z, label
    
    def make_transformers(self, is_train):
        if is_train:
            transformer = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.4),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomResizedCrop(self.image_size,scale=(0.8, 1.0)),
                transforms.ToTensor()
            ])
        else:
            transformer = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size,scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor()
            ])
        return transformer
