import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import glob
import os
from PIL import Image

class PMDataset(Dataset):
    def __init__(self, unimodal_datapaths):
        """
        Modified from https://github.com/thomassutter/MoPoE/blob/main/mmnist/MMNISTDataset.py
        """
        self.num_modalities = len(unimodal_datapaths)
        self.unimodal_datapaths = unimodal_datapaths
        self.transform = torchvision.transforms.ToTensor()
        # self.target_transform = target_transform

        # save all paths to individual files
        self.file_paths = {dp: [] for dp in self.unimodal_datapaths}
        for dp in self.unimodal_datapaths:
            files = glob.glob(os.path.join(dp, "*.png"))
            self.file_paths[dp] = files
        # assert that each modality has the same number of images
        num_files = len(self.file_paths[dp])
        for files in self.file_paths.values():
            assert len(files) == num_files
        self.num_files = num_files

        self.data_list = [torch.zeros(self.num_files,3,28,28) for i in range(self.num_modalities)]
        self.label_list = torch.zeros(self.num_files, dtype=int)
        
        for i in range(self.num_files):
            for dp_ind, dp in enumerate(self.unimodal_datapaths):
                self.data_list[dp_ind][i] = self.transform(Image.open(self.file_paths[dp][i]))
            self.label_list[i] = int(self.file_paths[dp][i].split(".")[-2])


    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        # files = [self.file_paths[dp][index] for dp in self.unimodal_datapaths]
        # images = [Image.open(files[m]) for m in range(self.num_modalities)]
        # labels = [int(files[m].split(".")[-2]) for m in range(self.num_modalities)]

        # # transforms
        # if self.transform:
        #     images = [self.transform(img) for img in images]

        # images_dict = {"m%d" % m: images[m] for m in range(self.num_modalities)}
        # return images_dict, labels[0]

        images_dict = {"m%d" % m: self.data_list[m][index] for m in range(self.num_modalities)}
        return images_dict, self.label_list[index].item()

class PM32Dataset(Dataset):
    def __init__(self, unimodal_datapaths):
        """
        Modified from https://github.com/thomassutter/MoPoE/blob/main/mmnist/MMNISTDataset.py
        """
        self.num_modalities = len(unimodal_datapaths)
        self.unimodal_datapaths = unimodal_datapaths
        self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Pad(2)])
        # self.target_transform = target_transform

        # save all paths to individual files
        self.file_paths = {dp: [] for dp in self.unimodal_datapaths}
        for dp in self.unimodal_datapaths:
            files = glob.glob(os.path.join(dp, "*.png"))
            self.file_paths[dp] = files
        # assert that each modality has the same number of images
        num_files = len(self.file_paths[dp])
        for files in self.file_paths.values():
            assert len(files) == num_files
        self.num_files = num_files

        self.data_list = [torch.zeros(self.num_files,3,32,32) for i in range(self.num_modalities)]
        self.label_list = torch.zeros(self.num_files, dtype=int)
        
        for i in range(self.num_files):
            for dp_ind, dp in enumerate(self.unimodal_datapaths):
                self.data_list[dp_ind][i] = self.transform(Image.open(self.file_paths[dp][i]))
            self.label_list[i] = int(self.file_paths[dp][i].split(".")[-2])


    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        # files = [self.file_paths[dp][index] for dp in self.unimodal_datapaths]
        # images = [Image.open(files[m]) for m in range(self.num_modalities)]
        # labels = [int(files[m].split(".")[-2]) for m in range(self.num_modalities)]

        # # transforms
        # if self.transform:
        #     images = [self.transform(img) for img in images]

        # images_dict = {"m%d" % m: images[m] for m in range(self.num_modalities)}
        # return images_dict, labels[0]

        images_dict = {"m%d" % m: self.data_list[m][index] for m in range(self.num_modalities)}
        return images_dict, self.label_list[index].item()

# Added to replicate mmplus
class PM28Dataset(Dataset):
    def __init__(self, unimodal_datapaths):
        """
        Modified from https://github.com/thomassutter/MoPoE/blob/main/mmnist/MMNISTDataset.py
        """
        self.num_modalities = len(unimodal_datapaths)
        self.unimodal_datapaths = unimodal_datapaths
        self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            ])
        # self.target_transform = target_transform

        # save all paths to individual files
        self.file_paths = {dp: [] for dp in self.unimodal_datapaths}
        for dp in self.unimodal_datapaths:
            files = glob.glob(os.path.join(dp, "*.png"))
            self.file_paths[dp] = files
        # assert that each modality has the same number of images
        num_files = len(self.file_paths[dp])
        for files in self.file_paths.values():
            assert len(files) == num_files
        self.num_files = num_files

        self.data_list = [torch.zeros(self.num_files,3,28,28) for i in range(self.num_modalities)]
        self.label_list = torch.zeros(self.num_files, dtype=int)
        
        for i in range(self.num_files):
            for dp_ind, dp in enumerate(self.unimodal_datapaths):
                self.data_list[dp_ind][i] = self.transform(Image.open(self.file_paths[dp][i]))
            self.label_list[i] = int(self.file_paths[dp][i].split(".")[-2])


    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        # files = [self.file_paths[dp][index] for dp in self.unimodal_datapaths]
        # images = [Image.open(files[m]) for m in range(self.num_modalities)]
        # labels = [int(files[m].split(".")[-2]) for m in range(self.num_modalities)]

        # # transforms
        # if self.transform:
        #     images = [self.transform(img) for img in images]

        # images_dict = {"m%d" % m: images[m] for m in range(self.num_modalities)}
        # return images_dict, labels[0]

        images_dict = {"m%d" % m: self.data_list[m][index] for m in range(self.num_modalities)}
        return images_dict, self.label_list[index].item()

def get_train_test_dataset():
    data_str_train = "./data/MMNIST/train"
    data_str_test = "./data/MMNIST/test"
    paired_train_dataset = PMDataset([data_str_train + '/m0', data_str_train + '/m1', data_str_train + '/m2', data_str_train + '/m3', data_str_train + '/m4'])
    paired_test_dataset = PMDataset([data_str_test + '/m0', data_str_test + '/m1', data_str_test + '/m2', data_str_test + '/m3', data_str_test + '/m4'])
    return paired_train_dataset, paired_test_dataset

def get_train_test_dataset_upd():
    data_str_train = "./data/UpdMMNIST/train"
    data_str_val = "./data/UpdMMNIST/val"
    data_str_test = "./data/UpdMMNIST/test"
    paired_train_dataset = PMDataset([data_str_train + '/m0', data_str_train + '/m1', data_str_train + '/m2', \
        data_str_train + '/m3', data_str_train + '/m4', data_str_train + '/m5', data_str_train + '/m6', \
        data_str_train + '/m7', data_str_train + '/m8', data_str_train + '/m9'])
    paired_val_dataset = PMDataset([data_str_val + '/m0', data_str_val + '/m1', data_str_val + '/m2', \
        data_str_val + '/m3', data_str_val + '/m4', data_str_val + '/m5', data_str_val + '/m6', \
        data_str_val + '/m7', data_str_val + '/m8', data_str_val + '/m9'])
    paired_test_dataset = PMDataset([data_str_test + '/m0', data_str_test + '/m1', data_str_test + '/m2', \
        data_str_test + '/m3', data_str_test + '/m4', data_str_test + '/m5', data_str_test + '/m6', \
        data_str_test + '/m7', data_str_test + '/m8', data_str_test + '/m9'])
    return paired_train_dataset, paired_val_dataset, paired_test_dataset

def get_train_test_dataset_upd10():
    data_str_train = "./data/Upd10MMNIST/train"
    data_str_val = "./data/Upd10MMNIST/val"
    data_str_test = "./data/Upd10MMNIST/test"
    paired_train_dataset = PMDataset([data_str_train + '/m0', data_str_train + '/m1', data_str_train + '/m2', \
        data_str_train + '/m3', data_str_train + '/m4', data_str_train + '/m5', data_str_train + '/m6', \
        data_str_train + '/m7', data_str_train + '/m8', data_str_train + '/m9'])
    paired_val_dataset = PMDataset([data_str_val + '/m0', data_str_val + '/m1', data_str_val + '/m2', \
        data_str_val + '/m3', data_str_val + '/m4', data_str_val + '/m5', data_str_val + '/m6', \
        data_str_val + '/m7', data_str_val + '/m8', data_str_val + '/m9'])
    paired_test_dataset = PMDataset([data_str_test + '/m0', data_str_test + '/m1', data_str_test + '/m2', \
        data_str_test + '/m3', data_str_test + '/m4', data_str_test + '/m5', data_str_test + '/m6', \
        data_str_test + '/m7', data_str_test + '/m8', data_str_test + '/m9'])
    return paired_train_dataset, paired_val_dataset, paired_test_dataset

def get_train_test_dataset_upd10_32x32():
    data_str_train = "./data/Upd10MMNIST/train"
    data_str_val = "./data/Upd10MMNIST/val"
    data_str_test = "./data/Upd10MMNIST/test"
    paired_train_dataset = PM32Dataset([data_str_train + '/m0', data_str_train + '/m1', data_str_train + '/m2', \
        data_str_train + '/m3', data_str_train + '/m4', data_str_train + '/m5', data_str_train + '/m6', \
        data_str_train + '/m7', data_str_train + '/m8', data_str_train + '/m9'])
    paired_val_dataset = PM32Dataset([data_str_val + '/m0', data_str_val + '/m1', data_str_val + '/m2', \
        data_str_val + '/m3', data_str_val + '/m4', data_str_val + '/m5', data_str_val + '/m6', \
        data_str_val + '/m7', data_str_val + '/m8', data_str_val + '/m9'])
    paired_test_dataset = PM32Dataset([data_str_test + '/m0', data_str_test + '/m1', data_str_test + '/m2', \
        data_str_test + '/m3', data_str_test + '/m4', data_str_test + '/m5', data_str_test + '/m6', \
        data_str_test + '/m7', data_str_test + '/m8', data_str_test + '/m9'])
    return paired_train_dataset, paired_val_dataset, paired_test_dataset

def get_train_test_dataset_upd10_28x28():
    data_str_train = "./data/Upd10MMNIST/train"
    data_str_val = "./data/Upd10MMNIST/val"
    data_str_test = "./data/Upd10MMNIST/test"
    paired_train_dataset = PM28Dataset([data_str_train + '/m0', data_str_train + '/m1', data_str_train + '/m2', \
        data_str_train + '/m3', data_str_train + '/m4', data_str_train + '/m5', data_str_train + '/m6', \
        data_str_train + '/m7', data_str_train + '/m8', data_str_train + '/m9'])
    paired_val_dataset = PM28Dataset([data_str_val + '/m0', data_str_val + '/m1', data_str_val + '/m2', \
        data_str_val + '/m3', data_str_val + '/m4', data_str_val + '/m5', data_str_val + '/m6', \
        data_str_val + '/m7', data_str_val + '/m8', data_str_val + '/m9'])
    paired_test_dataset = PM28Dataset([data_str_test + '/m0', data_str_test + '/m1', data_str_test + '/m2', \
        data_str_test + '/m3', data_str_test + '/m4', data_str_test + '/m5', data_str_test + '/m6', \
        data_str_test + '/m7', data_str_test + '/m8', data_str_test + '/m9'])
    return paired_train_dataset, paired_val_dataset, paired_test_dataset

def test_dataset_upd10_32x32(test=True):
    # data_str_train = "./data/Upd10MMNIST/train"
    # data_str_val = "./data/Upd10MMNIST/val"
    if test:
        data_str_test = "./data/Upd10MMNIST/test"
        paired_test_dataset = PM32Dataset([data_str_test + '/m0', data_str_test + '/m1', data_str_test + '/m2', \
        data_str_test + '/m3', data_str_test + '/m4', data_str_test + '/m5', data_str_test + '/m6', \
        data_str_test + '/m7', data_str_test + '/m8', data_str_test + '/m9'])
        return paired_test_dataset
    else:
        data_str_val = "./data/Upd10MMNIST/val"
        paired_val_dataset = PM32Dataset([data_str_val + '/m0', data_str_val + '/m1', data_str_val + '/m2', \
            data_str_val + '/m3', data_str_val + '/m4', data_str_val + '/m5', data_str_val + '/m6', \
            data_str_val + '/m7', data_str_val + '/m8', data_str_val + '/m9'])
        return paired_val_dataset
    
def test_dataset_upd10_28x28(test=True):
    # data_str_train = "./data/Upd10MMNIST/train"
    # data_str_val = "./data/Upd10MMNIST/val"
    if test:
        data_str_test = "./data/Upd10MMNIST/test"
        paired_test_dataset = PM28Dataset([data_str_test + '/m0', data_str_test + '/m1', data_str_test + '/m2', \
        data_str_test + '/m3', data_str_test + '/m4', data_str_test + '/m5', data_str_test + '/m6', \
        data_str_test + '/m7', data_str_test + '/m8', data_str_test + '/m9'])
        return paired_test_dataset
    else:
        data_str_val = "./data/Upd10MMNIST/val"
        paired_val_dataset = PM28Dataset([data_str_val + '/m0', data_str_val + '/m1', data_str_val + '/m2', \
            data_str_val + '/m3', data_str_val + '/m4', data_str_val + '/m5', data_str_val + '/m6', \
            data_str_val + '/m7', data_str_val + '/m8', data_str_val + '/m9'])
        return paired_val_dataset
    
def get_train_test_dataset2():
    data_str_train = "../data/MMNIST/train"
    data_str_test = "../data/MMNIST/test"
    paired_train_dataset = PMDataset([data_str_train + '/m0', data_str_train + '/m1', data_str_train + '/m2', data_str_train + '/m3', data_str_train + '/m4'])
    paired_test_dataset = PMDataset([data_str_test + '/m0', data_str_test + '/m1', data_str_test + '/m2', data_str_test + '/m3', data_str_test + '/m4'])
    return paired_train_dataset, paired_test_dataset