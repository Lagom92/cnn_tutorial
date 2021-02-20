import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision

from func import KneeDataset
from model import UNet

# Check GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: ", device)

batch_size_test = 1
lr = 0.001
momentum = 0.9

base_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

test_dataset = KneeDataset('./data/test_knee/', transform=base_transform)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

# Load model
net = UNet().to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

ckpt = torch.load('best_unet.pt', map_location=torch.device('cpu'))
net.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])

with torch.no_grad():
    net.eval()
    i = 0
    for img, mask in test_data_loader:
        inputs = img.to(device)
        labels = mask.to(device)
        outputs = net(inputs)

        plt.imsave(f"./result/{i}_inputs.png", inputs.squeeze(), cmap='gray')
        plt.imsave(f"./result/{i}_labels.png", labels.squeeze(), cmap='gray')
        plt.imsave(f"./result/{i}_pred.png", outputs.squeeze(), cmap='gray')

        i += 1
        
print("Success save")