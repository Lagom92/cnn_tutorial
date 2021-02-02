import os
import torch
import torchvision
import torchvision.transforms as transforms
from vgg16 import VGGNet

# GPU 지정 및 확인
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: ", device)

# Parameters
batch_size = 4
epochs = 3

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
    )

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True, num_workers=2
    )

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 저장한 모델 불러오기
vgg16 = VGGNet(10).to(device)
PATH = 'vgg16.pth'
vgg16.load_state_dict(torch.load(PATH))

# 전체 test 데이터셋에 대한 정확도
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f"Total Accuracy: {100 * correct / total}%")

# 어떤 class를 잘 분류했는지 확인
n = len(classes)

class_correct = [0.0]*n
class_total = [0.0]*n

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            
for i in range(n):
    print(f"Accuracy of {classes[i]}: {100*class_correct[i] / class_total[i]}%")