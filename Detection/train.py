import os
import torch
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
import albumentations.pytorch
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

Augmented_transform = A.Compose(
    [A.HorizontalFlip(p=1),
     A.Rotate(p=1),
     A.OneOf([
         A.ChannelShuffle(p=1),
         A.RGBShift(p=1)
     ], p=1),
     A.OneOf([
        A.MotionBlur(p=1),
        A.GaussNoise(p=1)
    ], p=1),
     A.pytorch.transforms.ToTensor()],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
)

dataset = func.MaskDataset('data/images/', transform=base_transform)

Augmented_dataset = func.AugmentedMaskDaset(
    path = 'data/images/',
    transform = Augmented_transform
)

train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=func.collate_fn)

# train_data_loader = torch.utils.data.DataLoader(Augmented_dataset, batch_size=batch_size, collate_fn=func.collate_fn)

# Load pretrained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

# Train
model.to(device)
    
# parameters
params = [p for p in model.parameters() if p.requires_grad] # gradient calculation이 필요한 params만 추출
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# len_train_data_loader = len(train_data_loader)
print("----Strart training----")
for epoch in range(epochs):
    model.train()
    i = 0    
    epoch_loss = 0
    for images, targets in train_data_loader:
        i += 1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets) 

        losses = sum(loss for loss in loss_dict.values()) 

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses 

    print(f"epoch: {epoch+1}/{epochs}, Loss: {epoch_loss}")

# Save model
torch.save(model.state_dict(), f'Augmented_model.pt')
