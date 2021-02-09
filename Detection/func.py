import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset
from PIL import Image

def generate_box(obj):
    xmin = float(obj.find('xmin').text)
    ymin = float(obj.find('ymin').text)
    xmax = float(obj.find('xmax').text)
    ymax = float(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    name = obj.find('name').text
    if name == "with_mask":
        return 1
    elif name == "mask_weared_incorrect":
        return 2
    return 0    # without_mask

def generate_target(file):
    with open(file) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        objects = soup.find_all('object')
        num_objs = len(objects)

        boxes, labels = [], []
        for obj in objects:
            boxes.append(generate_box(obj))
            labels.append(generate_label(obj))
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32) 
        labels = torch.as_tensor(labels, dtype=torch.int64) 

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return target

def plot_image(img_path, annotation):
    
    img = img.cpu().permute(1,2,0)
    
    fig,ax = plt.subplots(1)
    ax.imshow(img)

    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 0 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        elif annotation['labels'][idx] == 1 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
        else:
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')

        ax.add_patch(rect)

    plt.show()

class MaskDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.imgs = list(sorted(os.listdir(self.path)))
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        file_image = self.imgs[idx]
        file_label = self.imgs[idx][:-3] + 'xml'
        img_path = os.path.join(self.path, file_image)
        
        label_path = os.path.join("data/annotations/", file_label)

        img = Image.open(img_path).convert("RGB")
        target = generate_target(label_path)
        
        # to_tensor = torchvision.transforms.ToTensor()

        if self.transform:
            img = self.transform(img)
            # img, transform_target = self.transform(np.array(img), np.array(target['boxes']))
            # target['boxes'] = torch.as_tensor(transform_target)
        else:
            to_tensor = torchvision.transforms.ToTensor()
            img = to_tensor(img)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))