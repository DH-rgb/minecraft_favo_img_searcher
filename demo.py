import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models
from torchvision.utils import save_image

from utils import MyDataset
from model import MyNet

import os
import sys 
import argparse
import pdb


def main(args):
    #hyper parameter
    end_epoch = 200
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
    model.load_state_dict(torch.load("models/"+args.model+"/"+str(args.epoch-1)+".pth"))
    model.eval()

    criteria = nn.CrossEntropyLoss()

    dataset = MyDataset("demo/",is_train=False, is_demo=True)
    test_loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)

    with torch.no_grad():
        if not os.path.exists("demo/"+args.model+"/favo/"):
            os.makedirs("demo/"+args.model+"/favo/")
            os.makedirs("demo/"+args.model+"/no_favo/")
        for i, data in enumerate(test_loader):
            print("\riteration: {}".format(i),end="")
            inputs, labels = data
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs.data,1)

            favo_preds = preds.to('cpu')
            if favo_preds==1:
                save_image(inputs, "demo/"+args.model+"/favo/" +str(i) + ".jpg")
            elif favo_preds==0:
                save_image(inputs, "demo/"+args.model+"/no_favo/" +str(i) + ".jpg")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model","-m",help="model name")
    parser.add_argument("--epoch","-e",type=int,help="model epoch")
    args = parser.parse_args()
    main(args)