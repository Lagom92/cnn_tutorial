import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision


class KneeDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        
        lst_all_data = os.listdir(self.path)
                
        lst_data = []
        lst_mask = []
        
        for data in lst_all_data:
            if 'mask' in data:
                lst_mask.append(data)
            else:
                lst_data.append(data)
                
        lst_data.sort()
        lst_mask.sort()
        
        self.lst_data = lst_data
        self.lst_mask = lst_mask
        
    def __len__(self):
        return len(self.lst_data)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.lst_data[idx])
        mask_path = os.path.join(self.path, self.lst_mask[idx])
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        img = np.asarray(img, np.float32)
        img /= 255.0
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_CUBIC)
        mask = np.asarray(mask, np.float32)
        mask /= 255.0

        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return img, mask