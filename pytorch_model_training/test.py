from typing import List
import os
import time
import random
import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torchvision.transforms.v2 as v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def model_loader(feature_extracting:bool=True, num_classes:int=3, greyscale_single_channel:bool=True, imgsize = (1024,1024)):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    if greyscale_single_channel:
        # update transformations
        image_mean = [0.485]
        image_std = [0.229]
        transforms = GeneralizedRCNNTransform(imgsize[0],imgsize[1],image_mean,image_std)
        model.transform = transforms
        model.backbone.body.conv1 = torch.nn.Conv2d(1, 64,
                            kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False).requires_grad_(True)
        model.backbone.body.conv1.weight.data = model.backbone.body.conv1.weight.data.sum(axis=1).reshape(64, 1, 7, 7)
        print("Updated GeneralisedRCNNTransform and conv1 to support greyscale (1 channel) images")
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False # freeze all layers
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)  # these layers will always be trained
    if feature_extracting:
        print("Backbone frozen, tuning:")
    else:
        print("Fine tuning all layers:")   
    for name,param in model.named_parameters():
        if param.requires_grad:
            print("  ",name)     
    return model

def generate_transformations(extra_transforms:List=None, no_img_channels:int=1):
    base_transform_list = []
    base_transform_list.extend([v2.ToImage(),])
    base_transform_list.append(v2.Grayscale(num_output_channels=no_img_channels))
    if extra_transforms:
        base_transform_list.extend(extra_transforms)
    base_transform_list.extend([v2.SanitizeBoundingBoxes(),
                                v2.ToDtype(torch.float32, scale=True),])  # we always want this to be last transformation
    transforms = v2.Compose(base_transform_list)
    return transforms

def read_coco_dataset(images_root: str, annotations_path:str, transforms=None):
    dataset = torchvision.datasets.CocoDetection(root=images_root, annFile=annotations_path, transforms=transforms)
    dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("boxes", "labels"))
    return dataset

def fix_seed(seed):
    '''
    Args : 
        seed : fix the seed
    Function which allows to fix all the seed and get reproducible results
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

if __name__ == "__main__":
    model = model_loader(feature_extracting=True, num_classes=3, greyscale_single_channel=True, imgsize=(1024,1024))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model.to(device)
    # if greyscale = False, then we need to convert the greyscale images to not have 3 channels
    transforms = generate_transformations(no_img_channels=1)
    print(transforms) 
    """train_dataset = read_coco_dataset(images_root="/Users/kaichenchai/Documents/Projects/cardiomegaly_subset/train",
                                      annotations_path="subset/c-subset/explainability/cardiomegaly_400_subset_coco_labels.json",
                                      transforms=transforms)
    val_dataset = read_coco_dataset(images_root="/Users/kaichenchai/Documents/Projects/cardiomegaly_subset/val",
                                   annotations_path="subset/c-subset/explainability/cardiomegaly_100_subset_coco_labels_val.json",
                                   transforms=transforms)"""
    train_dataset = read_coco_dataset(images_root="/mnt/data/kai/VinDr_Subsets/cardiomegaly_subset/train",
                                      annotations_path="../subset/c-subset/explainability/cardiomegaly_400_subset_coco_labels.json",
                                      transforms=transforms)
    val_dataset = read_coco_dataset(images_root="/mnt/data/kai/VinDr_Subsets/cardiomegaly_subset/val",
                                   annotations_path="../subset/c-subset/explainability/cardiomegaly_100_subset_coco_labels_val.json",
                                   transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                                   collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, 
                                   collate_fn=lambda x: tuple(zip(*x)))
    
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(learnable_params, lr=0.005,
                                weight_decay=0.0005)
    
    num_epochs = 5
    
    print('----------------------train start--------------------------')
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        i = 0    
        epoch_loss = 0
        for imgs, annotations in tqdm.tqdm(train_loader):
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations) 
            print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())        

            optimizer.zero_grad()
            losses.backward()
            optimizer.step() 
            epoch_loss += losses
        print(f'epoch : {epoch+1}, Loss : {epoch_loss}, time : {time.time() - start}')
