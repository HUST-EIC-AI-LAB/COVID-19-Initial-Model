
import data 
import numpy as np 
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import model 
from data import TrainDataset, TestDataset
from model import densenet3d
from torch.nn import DataParallel
from logger import *
from test_case import test_case  
import logging

from warmup_scheduler import GradualWarmupScheduler

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


def train(train_data_loader, test_data_loader, model, optimizer, get_lr, log, num_epochs = 101, save_interval = 5, save_folder = "./checkpoint/"):
    train_num = len(train_data_loader)
    print("start training")
    recall_two_max = 0

    for epoch in range(num_epochs):
        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # for train
        running_loss = 0.0
        model.train()

        for index, (inputs, labels, patient_name) in enumerate(train_data_loader):
            optimizer.zero_grad()
            
            inputs, labels = inputs.cuda(), labels.cuda()#to(device), labels.to(device)
            for label in labels:
                if label.size() == torch.Size([2]):#== torch.tensor([[1, 1]]):
                    label = label[0]
                    print(patient_name)
            # foward
            inputs = inputs.unsqueeze(dim = 1).float()
            inputs = F.interpolate(inputs, size = [16, 128, 128], mode = "trilinear", align_corners = False)
            # print(inputs.shape,labels.shape)
            outputs = model(inputs)

            # backward
            loss = criterion(outputs, labels)
            loss.backward()
            #optimize
            optimizer.step()

            # loss update
            running_loss += loss.item()
     
            print("{} iter, loss {}".format(index + 1, loss.item()))


        print("{} epoch, loss {}".format(epoch + 1, running_loss / train_num))
        log.info("{} epoch, loss {}".format(epoch + 1, running_loss / train_num))

        running_loss = 0.0
        recall_two = test_case(test_data_loader, model)
        if recall_two > recall_two_max:
            recall_two_max = recall_two

        PATH = "checkpoint/{}_epoch_{}.pth".format(epoch, recall_two_max)
        log.info("save {} epoch.pth".format(epoch))
        torch.save(model.state_dict(), PATH)

        if epoch % 5 == 0:
            PATH = "checkpoint/{}_epoch.pth".format(epoch)
            log.info("save {} epoch.pth".format(epoch))
            torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    model = densenet3d().cuda()
    weight = torch.from_numpy(np.array([[0.2, 0.2, 0.4, 0.2]])).float()   

    criterion = nn.CrossEntropyLoss(weight=weight).cuda()
    lr_rate = 0.01
    optimizer = optim.SGD(model.parameters(),lr=lr_rate, momentum=0.9)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler_cosine)
    def get_lr(epoch):
        if epoch <= 10:
            lr = lr_rate
        elif epoch <= 20:
            lr = 0.1 * lr_rate
        else:
            lr = 0.01 * lr_rate
        return lr

    logfile = "./train_valid.log"
    sys.stdout = Logger(logfile)

    # resume_path = "./checkpoint/20_epoch.pth"
    # if resume_path:
    #     if os.path.isfile(resume_path):
    #         print("=> loading checkpoint '{}'".format(resume_path))
    #         checkpoint = torch.load(resume_path)
    #         model.load_state_dict(checkpoint)
           
    # sets.phase = 'train'
    train_data_train = TrainDataset()
    train_data_loader = DataLoader(dataset = train_data_train, batch_size = 4, shuffle = True, num_workers = 12)
    test_data_test = TestDataset()
    test_data_loader = DataLoader(dataset = test_data_test, batch_size = 2, shuffle = False, num_workers = 12)

    num_epochs = 101
    log = logging.getLogger()
    train(train_data_loader, test_data_loader, model, optimizer, get_lr, log)

    




