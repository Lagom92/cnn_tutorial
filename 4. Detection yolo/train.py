import os
# import numpy as np
# import random
from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.datasets as data
from utils import cosine_lr_scheduler
# import utils.data_augment as dataAug
# import utils.tools as tools
import config.yolov4_config as cfg

# import glob
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
import torch
from torch.utils.data import Dataset
from PIL import Image
# import torchvision
# import cv2

# import pandas as pd

from dataset import BuildDataset

def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), targets

# GPU device
# Check GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: ", device)

start_epoch = 0
best_mAP = 0.0
epochs = 1
eval_epoch = 2
batch_size = 1
weight_path = 'weight/mobilenetv2.pth'

# train_anno_path = './data/train_annotation.txt'
train_anno_path = './data/resized_train_annotation.txt'

train_dataset = BuildDataset(train_anno_path)

train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
        )

# model
yolov4 = Build_Model(weight_path=weight_path).to(device)

optimizer = optim.SGD(
            yolov4.parameters(),
            lr=cfg.TRAIN["LR_INIT"],
            momentum=cfg.TRAIN["MOMENTUM"],
            weight_decay=cfg.TRAIN["WEIGHT_DECAY"],
        )

criterion = YoloV4Loss(
            anchors=cfg.MODEL["ANCHORS"],
            strides=cfg.MODEL["STRIDES"],
            iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"],
        )

scheduler = cosine_lr_scheduler.CosineDecayLR(
            optimizer,
            T_max=epochs * len(train_dataloader),
            lr_init=cfg.TRAIN["LR_INIT"],
            lr_min=cfg.TRAIN["LR_END"],
            warmup=cfg.TRAIN["WARMUP_EPOCHS"] * len(train_dataloader),
        )


# Training
for epoch in range(start_epoch, epochs):
    yolov4.train()
    
    mloss = torch.zeros(4)
    for i, data in enumerate(train_dataloader):
        scheduler.step(
                    len(train_dataloader)
                    / (cfg.TRAIN["BATCH_SIZE"])
                    * epoch
                    + i
                )
        
        imgs = data[0]
        label_sbbox = data[1]
        label_mbbox = data[2]
        label_lbbox = data[3]
        sbboxes = data[4]
        mbboxes = data[5]
        lbboxes = data[6]
        
        imgs = imgs.to(device)
        label_sbbox = label_sbbox.to(device)
        label_mbbox = label_mbbox.to(device)
        label_lbbox = label_lbbox.to(device)
        sbboxes = sbboxes.to(device)
        mbboxes = mbboxes.to(device)
        lbboxes = lbboxes.to(device)
        
        p, p_d = yolov4(imgs)

        loss, loss_ciou, loss_conf, loss_cls = criterion(
            p,
            p_d,
            label_sbbox,
            label_mbbox,
            label_lbbox,
            sbboxes,
            mbboxes,
            lbboxes,
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_items = torch.tensor([loss_ciou, loss_conf, loss_cls, loss])
        mloss = (mloss * i + loss_items) / (i + 1)
        
        if i%10 == 0:
            print(f"Epoch: {epoch}/{epochs}, step: [{i}/{len(train_dataloader) - 1}], mloss: {mloss}")
#             print(f"=== Epoch:[{epoch}/{epochs}], step:[{i}/{len(self.train_dataloader) - 1}], img_size:[{train_dataset.img_size:3}], total_loss:{mloss[3]:.4f}|loss_ciou:{mloss[0]:.4f}|loss_conf:{mloss[1]:.4f}|loss_cls:{mloss[2]:.4f}|lr:{optimizer.param_groups[0]["lr"]:.6f}")
        
        chkpt = {
            "epoch": epoch,
            # "best_mAP": self.best_mAP,
            "model": yolov4.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(chkpt, './weight/model.pt')