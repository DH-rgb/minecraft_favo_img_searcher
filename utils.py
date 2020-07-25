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
            # dir_favo = os.path.join(datadir,"train_favo")
            # dir_no_favo = os.path.join(datadir,"train_no_favo")
            self.image_path = [p for p in glob(datadir+"train/**/*.jpg")]
        else:
            # dir_favo = os.path.join(datadir,"test_favo")
            # dir_no_favo = os.path.join(datadir,"test_no_favo")
            self.image_path = [p for p in glob(datadir+"test/**/*.jpg")]
        # self.image_path = [p for p in glob(datadir+"/**/*.jpg")]
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



    

    # def make_shuffle_data(self, dir_favo, dir_no_favo):

    # def make_dataset(self, dir_favo, dir_no_favo):
    #     images = []
    #     for fname in os.listdir(dir_favo):
    #         if fname.endswith('.jpg'):
    #             path  = os.path.join(dir_favo, fname)
    #             images.append(path)
    #     for fname in os.listdir(dir_no_favo):
    #         if fname.endswith('.jpg'):
    #             path  = os.path.join(dir_no_favo, fname)
    #             images.append(path)
    #     sorted(images)
    #     return images
