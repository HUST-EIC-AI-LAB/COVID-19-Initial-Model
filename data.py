from torch.utils.data import Dataset
import torch 
import torch.nn as nn 
import os
import pandas as pd
import numpy as np
import random
import nibabel as nib
from torch.utils.data import DataLoader

def load_image(image_path, mean, std):
    image = nib.load(image_path).get_fdata()#.astype(np.int32)
    np.subtract(image, mean, out = image)
    np.divide(image, std, out = image)
    image = image.transpose(2, 1, 0)
    return image


class TrainDataset(Dataset):
    def __init__(self):
        self.data_dir = "/home/xdjf/jhyang/ncp/src/dataset/"
        train_df = pd.read_csv("./utils/train_clean_new.csv")
        self.names_train = train_df["name"]#["B19_PA11_SE1"]#
        self.labels_train_df = pd.read_csv("./utils/train_clean_new.csv", index_col=0)
       

    def __getitem__(self, item):
        margin = 8
        # print(margin)
        name_train = self.names_train[item]
        label_train = self.labels_train_df.at[name_train, "four_label"]
        path_train = self.data_dir + name_train + ".nii.gz"
        image_train = nib.load(path_train).get_fdata().astype(np.int32).transpose(2, 1, 0)
        z_train, h_train, w_train = image_train.shape
        index_list=[]
        if z_train<=80:  
            start=random.randrange(0,z_train-16)
            for i in range(16):
                index_list.append(start+i*0)
        elif z_train<=160:
            start=random.randrange(0,z_train-80)
            for i in range(16):
                index_list.append(start+i*5)
        elif:
            start=random.randrange(0,z_train-160) 
            for i in range(16):
                index_list.append(start*i+10) 

        image_train_crop=[]
        for index in index_list:
            image_train_crop.append(image_train[index,:,:])
        image_train_crop=torch.cat(image_train_crop,dim=0).float()
        
        
        #image_train_crop = image_train[(z_train//2 - margin) : (z_train//2 + margin), :, :]
        return image_train_crop, label_train, name_train
        
    
    def __len__(self): 
        return len(self.names_train)


class TestDataset(Dataset):
    def __init__(self):
        self.data_dir = "/home/xdjf/jhyang/ncp/src/dataset/"
        test_df = pd.read_csv("./utils/test_norm.csv")
        self.names_test = test_df["name"]
        self.labels_test_df = pd.read_csv("./utils/test_norm.csv", index_col=0)
    

    def __getitem__(self, item):
        margin = 8
        name_test = self.names_test[item]
        label_test = self.labels_test_df.at[name_test, "four_label"]
        patient_id = self.labels_test_df.at[name_test, "patient_id"]
        path_test = self.data_dir + name_test + ".nii.gz"
        image_test = nib.load(path_test).get_fdata().astype(np.int32).transpose(2, 1, 0)
        z_test, h_test, w_test = image_test.shape
        index_list=[]
        if z_train<=80:  
            start=random.randrange(0,z_train-16)
            for i in range(16):
                index_list.append(start+i*0)
        elif z_train<=160:
            start=random.randrange(0,z_train-80)
            for i in range(16):
                index_list.append(start+i*5)
        elif:
            start=random.randrange(0,z_train-160) 
            for i in range(16):
                index_list.append(start*i+10) 

        image_train_crop=[]
        for index in index_list:
            image_train_crop.append(image_train[index,:,:])
        image_train_crop=torch.cat(image_train_crop,dim=0).float()
        #image_test_crop = image_test[(z_test//2 - margin) : (z_test//2 + margin), :, :]
        return image_test_crop, label_test, name_test, patient_id

    
    def __len__(self):
        return len(self.names_test)
