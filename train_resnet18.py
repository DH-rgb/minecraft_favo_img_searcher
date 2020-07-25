import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models

from utils import ResNet_Dataset

import os
import argparse
import pdb


def main(args):
    #hyper parameter
    end_epoch = 200
    lr = 0.001
    beta1 = 0.5
    beta2 = 0.99
    gpu_num = 0


    #set model
    model_ft = models.resnet18(pretrained=True)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features,1)
    model_ft.to("cuda:{}".format(gpu_num))

    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=lr,betas=(beta1,beta2))



    dataset = ResNet_Dataset("data/",is_train=True)
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True,num_workers=4)
    pdb.set_trace()
    
    for epoch in range(end_epoch): 
        epoch_loss = 0
        epoch_acc = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_ft(inputs.to(device))
            _, preds = torch.max(outputs.data,1)
            loss = criteria(outputs,labels.to(device))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data[0].detach()*inputs.size(0)
            epoch_acc += torch.sum(preds==labels.data)
        
        epoch_loss /= len(train_loader)*4
        epoch_acc /= len(train_loader)*4

        print("{epoch}: loss={epoch_loss}, Acc={epoch_acc}")
        if epoch%10==0:
            if not os.path.exist("models/ResNet/"+args.model):
                os.makedirs("models/ResNet/"+args.model)
            torch.save(model_ft.state_dict(),"models/ResNet/"+args.model+"/"+str(epoch)+".pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model","-m",help="model name")
    args = parser.parse_args()
    main(args)