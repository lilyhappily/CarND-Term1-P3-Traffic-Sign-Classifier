#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch.utils.data import Dataset, sampler
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torchvision.transforms.functional as TF
import PIL
import pickle

class TrafficSignDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_folder, dataset = 'train', transform = None):
        """
        @ param data_folder: the folder of datasets.
        @ param dataset: tell we read train, valid or test dataset.
        @ transform: whether we use augmentation on dataset or not, here we only use it on training dataset.
        """
        # Read the dataset 
        assert dataset in {'train', 'valid', 'test'}
        self.data_folder = os.path.join(data_folder, dataset + '.p')
        
        with open(self.data_folder, mode='rb') as f:
            data = pickle.load(f)
            
        # The pickled data is a dictionary with 4 key/value pairs, We only use 'features' and 'labels'.
        self.features = data["features"]
        self.labels = torch.from_numpy(data["labels"]).to(dtype=torch.long)
        self.transform = transform
        
    def __getitem__(self, index):
        
        # Read a sign according to the index
        sign_image = self.features[index]
        
        if self.transform:
            sign_image = self.transform(sign_image)
            
        return (sign_image, self.labels[index])
    
    def __len__(self):
        return len(self.features)
    
    
        
def get_dataset_transform(dataset='train'):
    """
    Using Pytorch torchvision.transforms to augmentate dataset.
    """
    
    assert dataset in {'train', 'valid', 'test'}
    
    if dataset == 'train':
        transform = transforms.Compose([transforms.ToPILImage(),  # The following operation need a PIL image as an input
                                        # Select one of the following list with a given probability
                                        transforms.RandomApply([
                                           transforms.RandomRotation(20, resample = PIL.Image.BICUBIC),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomAffine(0, translate=(0.1, 0.2), resample = PIL.Image.BICUBIC),
                                           transforms.RandomAffine(0, shear=20, resample=PIL.Image.BICUBIC),
                                           transforms.RandomAffine(0, scale=(0.8, 1.2), resample=PIL.Image.BICUBIC)]),
                                         transforms.Grayscale(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, ), 
                                                              std=(1, ))])# Mean center data
                                                           
    
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, ), 
                                                         std=(1, ))])# Mean center data
    
    return transform     

