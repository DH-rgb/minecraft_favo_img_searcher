import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models, datasets, transforms
from torchvision.utils import save_image

from utils import MyDataset
from model import MyNet

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
    model = MyNet()

    #set GPU or CPU
    if gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(gpu)
    else:
        device = 'cpu'  
    model.to(device)

    #print params
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()         
    print(params)
    print(model)

    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(beta1,beta2))

    dataset = MyDataset("data/",is_train=True)
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True)
    
    for epoch in range(end_epoch): 
        epoch_loss = 0
        epoch_acc = 0
        for i, data in enumerate(train_loader):
            print("\repoch: {} iteration: {}".format(epoch,i),end="")
            
            inputs, labels = data 
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs.data,1)
            loss = criteria(outputs,labels.to(device))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.to('cpu')*inputs.size(0)
            epoch_acc += torch.sum(preds.to('cpu')==labels.data)
        
        epoch_loss /= len(train_loader)*4
        epoch_acc = epoch_acc/float(len(train_loader)*4)
        

        print("[epoch: {}] [Loss: {:.4f}] [Acc: {:.4f}]".format(epoch, epoch_loss, epoch_acc))
        if (epoch+1)%10==0:
            if not os.path.exists("models/"+args.model):
                os.makedirs("models/"+args.model)
            torch.save(model.state_dict(),"models/"+args.model+"/"+str(epoch)+".pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model","-m",help="model name")
    args = parser.parse_args()
    main(args)