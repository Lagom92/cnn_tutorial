import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.dataset import random_split
import math
from vgg16 import VGGNet

# Check GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: ", device)

# Parameters
batch_size = 16
epochs = 20
lr = 0.001
momentum = 0.9
split_ratio = 0.2

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
num = len(dataset)
trainset, valset = random_split(dataset, [int(num*(1-split_ratio)), int(num*split_ratio)])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Build model
vgg16 = VGGNet(10).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.parameters(), lr=lr, momentum=momentum)

# Training
print("Start training")
best_val_acc = 0.0
for epoch in range(epochs):
    # train part
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"{epoch+1}/{epochs} epoch, loss: {loss.item():.3f}")

    # validation part
    with torch.no_grad():
        correct, total = 0, 0
        for i, data in enumerate(valloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = vgg16(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        if val_acc > best_val_acc:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': vgg16.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
                }, 'vgg16_ckpt.pt')
            best_val_acc = val_acc
            print(f"Save new cp | val acc: {val_acc}, loss: {loss.item():.3f}")


# Save trained model
# PATH = './vgg16.pth'
# torch.save(vgg16.state_dict(), PATH)
print("Save model")