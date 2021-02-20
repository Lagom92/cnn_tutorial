import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from func import KneeDataset
from model import UNet

# Check GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: ", device)

batch_size_train = 8
batch_size_val = 2
num_epoch = 1
lr = 0.001
momentum = 0.9

base_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

train_dataset = KneeDataset('./data/train_knee/', transform=base_transform)
val_dataset = KneeDataset('./data/validation_knee/', transform=base_transform)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

# Load model
net = UNet().to(device)

# Loss function and Optimizer
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

# training
print("start training")
for epoch in range(1, num_epoch+1):
    net.train()
    epoch_loss = []
    for img, mask in train_data_loader:
        inputs = img.to(device)
        labels = mask.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += [loss.item()]
        
    print(f"train epoch: {epoch}/{num_epoch}, loss: {np.mean(epoch_loss):.3f}")

    with torch.no_grad():
        net.eval()
        epoch_loss = []

        for img, mask in val_data_loader:
            inputs = img.to(device)
            labels = mask.to(device)

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            epoch_loss += [loss.item()]

            print(f"val epoch: {epoch}/{num_epoch}, loss: {np.mean(epoch_loss):.3f}")

# Save model
torch.save(net.state_dict(), 'unet.pt')
