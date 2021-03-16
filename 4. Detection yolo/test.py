import os
import torch
import cv2
import matplotlib.pyplot as plt
from model.build_model import Build_Model
from utils.tools import *
from eval.evaluator import Evaluator
import config.yolov4_config as cfg
from utils.visualize import *
from utils.torch_utils import *

# GPU device
# Check GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: ", device)

model = Build_Model().to(device)

w_path = 'model.pt'
chkpt = torch.load(os.path.join(w_path))

model.load_state_dict(chkpt['model'])

classes = ['1', '2', '3', '4', '5', '6']

test_path = 'test_img/'
imgs_path = os.listdir(test_path)
ratio = 0.1

for img_path in imgs_path:
    path = os.path.join(test_path, img_path)
    print(path)
    
    img = cv2.imread(path)
    resized_img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
#     resized_img = cv2.resize(img, (832, 832), interpolation=cv2.INTER_AREA)

    bboxes_prd = Evaluator(model).get_bbox(resized_img, img_path)
    
    if bboxes_prd.shape[0] != 0:
        boxes = bboxes_prd[..., :4]
        class_inds = bboxes_prd[..., 5].astype(np.int32)
        scores = bboxes_prd[..., 4]

        result = visualize_boxes(
            image=resized_img,
            boxes=boxes,
            labels=class_inds,
            probs=scores,
            class_labels=classes,
        )
        cv2.imwrite(test_path + 'pred_' + img_path, result)
        print("Save Image")