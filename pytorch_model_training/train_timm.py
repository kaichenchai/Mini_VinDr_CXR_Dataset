from typing import List
import time
import tqdm
from copy import deepcopy
import os

import timm
import wandb
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torchvision.transforms.v2 as v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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

class ValidationLossEarlyStopping:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience  # number of times to allow for no improvement before stopping the execution
        self.min_delta = min_delta  # the minimum change to be counted as improvement
        self.counter = 0  # count the number of times the validation accuracy not improving
        self.min_validation_loss = np.inf

    # return True when validation loss is not decreased by the `min_delta` for `patience` times 
    def early_stop_check(self, validation_loss):
        if ((validation_loss+self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0  # reset the counter if validation loss decreased at least by min_delta
        elif ((validation_loss+self.min_delta) > self.min_validation_loss):
            self.counter += 1 # increase the counter if validation loss is not decreased by the min_delta
            if self.counter >= self.patience:
                return True
        return False


if __name__ == "__main__":
    model = timm.create_model("", pretrained=True, num_classes=2, in_chans=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)
    # if greyscale = False, then we need to convert the greyscale images to not have 3 channels
    transforms = generate_transformations(no_img_channels=1)
    print(transforms) 
    """train_dataset = read_coco_dataset(images_root="/Users/kaichenchai/Documents/Projects/cardiomegaly_subset/train",
                                      annotations_path="subset/c-subset/explainability/cardiomegaly_400_subset_coco_labels.json",
                                      transforms=transforms)
    val_dataset = read_coco_dataset(images_root="/Users/kaichenchai/Documents/Projects/cardiomegaly_subset/val",
                                   annotations_path="subset/c-subset/explainability/cardiomegaly_100_subset_coco_labels_val.json",
                                   transforms=transforms)
    
    train_dataset = read_coco_dataset(images_root="/mnt/data/kai/VinDr_Subsets/cardiomegaly_subset/train",
                                      annotations_path="../subset/c-subset/explainability/cardiomegaly_400_subset_coco_labels.json",
                                      transforms=transforms)
    val_dataset = read_coco_dataset(images_root="/mnt/data/kai/VinDr_Subsets/cardiomegaly_subset/val",
                                   annotations_path="../subset/c-subset/explainability/cardiomegaly_100_subset_coco_labels_val.json",
                                   transforms=transforms)"""

    train_dataset = read_coco_dataset(images_root="/mnt/data/kai/VinDr_datasets/cardiomegaly_subset/1024_padding_CLAHE/train",
                                      annotations_path="../subset/c-subset/explainability/cardiomegaly_400_subset_coco_labels.json",
                                      transforms=transforms)
    val_dataset = read_coco_dataset(images_root="/mnt/data/kai/VinDr_datasets/cardiomegaly_subset/1024_padding_CLAHE/val",
                                   annotations_path="../subset/c-subset/explainability/cardiomegaly_100_subset_coco_labels_val.json",
                                   transforms=transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                                   collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, 
                                   collate_fn=lambda x: tuple(zip(*x)))
    
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(learnable_params, lr=0.0001,
                                weight_decay=0.001)
    
    torch.manual_seed(42069)
    num_epochs = 500
    current_min_loss = np.inf
    
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
    early_stopper = ValidationLossEarlyStopping(patience=50, min_delta=0.01)
    
    print('----------------------train start--------------------------')

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
        if validation_loss < current_min_loss:
            best_model_state = deepcopy(model.state_dict())
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
        dict_to_log["train/avg_loss"] = train_loss
        dict_to_log["validate/avg_loss"] = validation_loss
        run.log(data=dict_to_log,
                step=epoch,
                commit=True)
        if early_stopper.early_stop_check(validation_loss):             
            break
    
    file_path_base = f"./outputs/{time.strftime('%Y%m%d_%H%M%S')}"
    if not(os.path.isdir(file_path_base)):
        os.mkdir(file_path_base)
    file_path = f"{file_path_base}/{num_epochs}e_model.pt"
    file_path_lowest_loss = f"{file_path_base}/{num_epochs}e_model_min_val_loss.pt"
    torch.save(model.state_dict(), file_path)
    torch.save(best_model_state, file_path_lowest_loss)
    run.save(file_path, base_path=file_path_base)
    run.save(file_path_lowest_loss, base_path=file_path_base)
