import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models

from utils import ResNet_Dataset

import os
import sys 
import argparse
import pdb


def main(args):
    #hyper parameter
    end_epoch = 100
    lr = 0.001
    beta1 = 0.5
    beta2 = 0.99
    gpu = 0


    #set model
    model_ft = models.resnet18(pretrained=True)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features,2)
    #set GPU or CPU
    if gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(gpu)
    else:
        device = 'cpu'
    model_ft.to(device)

    #print params
    params = 0
    for p in model_ft.parameters():
        if p.requires_grad:
            params += p.numel()     
    print(params)

    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=lr,betas=(beta1,beta2))

    dataset = ResNet_Dataset("data/",is_train=True)
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True)
    
    for epoch in range(end_epoch): 
        epoch_loss = 0
        epoch_acc = 0
        for i, data in enumerate(train_loader):
            print("\repoch: {} iteration: {}".format(epoch,i),end="")
            inputs, labels = data 
            optimizer.zero_grad()
            outputs = model_ft(inputs.to(device))
            _, preds = torch.max(outputs.data,1)
            loss = criteria(outputs,labels.to(device))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.to('cpu')*inputs.size(0)
            epoch_acc += torch.sum(preds.to('cpu')==labels.data)
        
        epoch_loss /= len(train_loader)*4
        epoch_acc = epoch_acc/float(len(train_loader)*4)
        

        print("[Loss: {}] [Acc: {}]".format(epoch_loss,epoch_acc))
        if epoch%10==0:
            if not os.path.exists("models/ResNet/"+args.model):
                os.makedirs("models/ResNet/"+args.model)
            torch.save(model_ft.state_dict(),"models/ResNet/"+args.model+"/"+str(epoch)+".pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model","-m",help="model name")
    args = parser.parse_args()
    main(args)