import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models
from torchvision.utils import save_image

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
    model_ft = models.resnet18()
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features,2)
    model_ft.load_state_dict(torch.load("models/"+args.model+"/"+str(args.epoch)+".pth"))

    #set GPU or CPU
    if gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(gpu)
    else:
        device = 'cpu'
    model_ft.to(device)

    criteria = nn.CrossEntropyLoss()

    dataset = ResNet_Dataset("data/",is_train=False)
    test_loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)

    with torch.no_grad():
        epoch_loss = 0
        epoch_acc = 0
        if not os.path.exists("result/"+args.model+"/true_favo/"):
            os.makedirs("result/"+args.model+"/true_favo/")
            os.makedirs("result/"+args.model+"/true_no_favo/")
            os.makedirs("result/"+args.model+"/false_favo/")
            os.makedirs("result/"+args.model+"/false_no_favo/")
        for i, data in enumerate(test_loader):
            print("\riteration: {}".format(i),end="")
            inputs, labels = data
            outputs = model_ft(inputs.to(device))
            _, preds = torch.max(outputs.data,1)
            loss = criteria(outputs,labels.to(device))

            favo_preds = preds.to('cpu')
            if favo_preds==labels:
                if favo_preds==1:
                    save_image(inputs, "result/"+args.model+"/true_favo/" +str(i) + ".jpg")
                elif favo_preds==0:
                    save_image(inputs, "result/"+args.model+"/true_no_favo/" +str(i) + ".jpg")
            else:
                if favo_preds==1:
                    save_image(inputs, "result/"+args.model+"/false_favo/" +str(i) + ".jpg")
                elif favo_preds==0:
                    save_image(inputs, "result/"+args.model+"/false_no_favo/" +str(i) + ".jpg")

            epoch_loss += loss.data.to('cpu')*inputs.size(0)
            epoch_acc += torch.sum(favo_preds==labels.data)

        epoch_loss /= len(test_loader)
        epoch_acc = epoch_acc/float(len(test_loader))
        print("[Loss: {}] [Acc: {}]".format(epoch_loss,epoch_acc))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model","-m",help="model name")
    parser.add_argument("--epoch","-e",type=int,help="model epoch")
    args = parser.parse_args()
    main(args)