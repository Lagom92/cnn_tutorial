import os
import torch
import torchvision
# from torchvision import transforms
import func

# Check GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: ", device)

batch_size = 4
epochs = 2


base_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
train_dataset = func.MaskDataset('data/images/', transform=base_transform)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=func.collate_fn)

# Load pretrained model
retina = torchvision.models.detection.retinanet_resnet50_fpn(num_classes = 3, pretrained=False, pretrained_backbone = True)

# Train
retina.to(device)
    
# parameters
# params = [p for p in retina.parameters() if p.requires_grad] # gradient calculation이 필요한 params만 추출
optimizer = torch.optim.SGD(retina.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# len_train_data_loader = len(train_data_loader)
print("Strart training")
for epoch in range(epochs):
    retina.train()
    i = 0    
    epoch_loss = 0
    for images, targets in train_data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = retina(images, targets) 

        losses = sum(loss for loss in loss_dict.values()) 

        i += 1

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses 
    print(epoch_loss)

