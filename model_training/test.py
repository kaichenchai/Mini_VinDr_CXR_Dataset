import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision import transforms
from torchvision import tv_tensors

import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def model_loader(device="cpu", num_classes:int=3):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)
    return model

def read_coco_dataset(images_root: str, annotations_path:str, transformations):
    dataset = torchvision.datasets.CocoDetection(root=images_root, annFile=annotations_path, transforms=transformations)
    return dataset

def 

if __name__ == "__main__":
    print(model_loader())