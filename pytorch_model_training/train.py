from typing import List, Union
import os
import time
import random
import tqdm


import wandb
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
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
    dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("image_id", "boxes", "labels"))
    return dataset


if __name__ == "__main__":
    model = model_loader(feature_extracting=True, num_classes=3, greyscale_single_channel=True, imgsize=(1024,1024))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                                   collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, 
                                   collate_fn=lambda x: tuple(zip(*x)))
    
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(learnable_params, lr=0.0001,
                                weight_decay=0.0005)
    
    num_epochs = 5
    #fix_seed(12451)
    
    run = wandb.init(project="cardiomegaly_explainability",  # Change this to your desired project name
        name=f"fasterrcnn_{time.strftime('%Y%m%d_%H%M%S')}",  # Optional: unique run name
        config={
            "model": "fasterrcnn_resnet50_fpn_v2",
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": "AdamW",
            "learning_rate": optimizer.param_groups[0]["lr"]
    })
    
    metrics = MeanAveragePrecision(iou_type="bbox", extended_summary=True)
    
    print('----------------------train start--------------------------')
    torch.manual_seed(123)
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        i = 0    
        train_loss = 0
        for imgs, annotations in tqdm.tqdm(train_loader):
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items() if k != "image_id"} for t in annotations]
            train_loss_dict = model(imgs, annotations) 
            losses = sum(loss for loss in train_loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step() 
            train_loss += losses
        print(f'(Train) epoch : {epoch}, Avg Loss : {train_loss/len(train_loader)}, time : {time.time() - start}')
        validation_loss = 0
        with torch.no_grad():
            for imgs, annotations in tqdm.tqdm(val_loader):
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items() if k != "image_id"} for t in annotations]
                val_loss_dict = model(imgs, annotations)
                losses = sum(loss for loss in val_loss_dict.values())
                validation_loss += losses
        print(f'(Val) epoch : {epoch}, Avg Loss : {validation_loss/len(val_loader)}')
        model.eval()
        with torch.no_grad():
            for imgs, annotations in val_loader:
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items() if k != "image_id"} for t in annotations]
                val_predictions = model(imgs, annotations)
                metrics.update(preds=val_predictions, target=annotations)
        validation_results_dict = metrics.compute()
        # loss logging
        dict_to_log = {}
        for key, value in train_loss_dict.items():
            dict_to_log[f"train/{key}"] = value
        for key, value in val_loss_dict.items():
            dict_to_log[f"validate/{key}"] = value
        dict_to_log["metrics/mAP50-95"] = validation_results_dict["map"]
        dict_to_log["metrics/mAP_50"] = validation_results_dict["map_50"]
        run.log(data=dict_to_log,
                step=epoch,
                commit=True)