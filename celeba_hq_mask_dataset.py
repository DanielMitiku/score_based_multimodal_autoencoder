import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
import torchvision
import glob
import os
from PIL import Image

class CelebAHQMaskDS(Dataset):
    def __init__(self, size=128, datapath='./ddata/data/CelebAMask-HQ/',  ds_type='train'):
        """
            Args: 
                datapath: folder path containing train, val, and test folders of images and mask and celeba attribute text file
                transform: torchvision transform for the images and masks
                ds_type: train, val, or test
        """

        super().__init__()
        self.size = size
        self.ds_type = ds_type
        self.datapath = datapath
        self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Resize(self.size)])

        self.img_files = glob.glob(os.path.join(self.datapath + self.ds_type + '_img', "*.jpg"))
        self.mask_files = glob.glob(os.path.join(self.datapath + self.ds_type + '_mask', "*.png"))
        self.img_files.sort()
        self.mask_files.sort()
        assert len(self.img_files) == len(self.mask_files)
        
        self.attr_tensor = torch.zeros((len(self.img_files),40), dtype=int)
        self.img_tensor = torch.zeros(len(self.img_files),3,self.size,self.size)
        self.mask_tensor = torch.zeros(len(self.img_files),1,self.size,self.size)
        
        # Read attr text file
        attr_txt_file = open(self.datapath + 'CelebAMask-HQ-attribute-anno.txt')
        attr_list = attr_txt_file.readlines()
        self.attributes = attr_list[1].strip().split(" ")
        assert len(self.attributes) == 40
        
        for i in range(len(self.img_files)):
            assert self.img_files[i].split("/")[-1][:-4] == self.mask_files[i].split("/")[-1][:-4]
            self.img_tensor[i] = self.transform(Image.open(self.img_files[i]))
            self.mask_tensor[i] = self.transform(Image.open(self.mask_files[i]))
            
            img_idx = int(self.img_files[i].split("/")[-1][:-4])
            attr_i = attr_list[img_idx + 2].strip().split(" ")
            assert img_idx == int(attr_i[0][:-4])
            attr_i01 = torch.tensor([1 if a == '1' else 0 for a in attr_i[2:]])
            self.attr_tensor[i] = attr_i01
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        Returns a tuple of image, mask, attribute
        """
        return (self.img_tensor[index], self.mask_tensor[index], self.attr_tensor[index])