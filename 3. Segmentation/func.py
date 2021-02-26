import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from PIL import Image

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
        
        img = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('1')
        
        img_resized = img.resize((512, 512))
        mask_resized = mask.resize((512, 512))
        
        # to array
        img_arr = np.array(img_resized, 'float32')
        mask_arr = np.array(mask_resized)
        
        img_arr = img_arr / 255.0
        mask_arr = mask_arr.astype(float)

        if img_arr.ndim == 2:
            img_arr = img_arr[:, :, np.newaxis]
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[:, :, np.newaxis]

        if self.transform:
            img_arr = self.transform(img_arr)
            mask_arr = self.transform(mask_arr)
        
        return img_arr, mask_arr
        