import numpy as np
from PIL import Image, ImageOps
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import ops

#if over iou, then take average of observations
#https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
#https://doi.org/10.1016/j.imavis.2021.104117 #implementation of WBF without the weights from the original paper
def mergeIOUAverage(annoCSV, threshold = 0.5, csvName = "mergedIOUAverage.csv"):
    try:
        #read in csv
        oldAnno = pd.read_csv(annoCSV, sep = ",")
    except FileNotFoundError:
        print("CSV not found!")
    else:
        oldLength = len(oldAnno)
        newAnnos = [] #empty list to hold new annotations once they have been created
        if "rad_id" in oldAnno.columns:
            oldAnno = oldAnno.drop("rad_id", axis = 1)
        oldAnno = oldAnno.dropna(axis = 0) #drop all rows with no findings
        oldAnno = oldAnno[["image_id","class_name","x_min","y_min","x_max","y_max"]]
        #groupby both image id and class
        #all the other stuff is for performance
        groups = oldAnno.groupby(["image_id", "class_name"], group_keys = False, as_index = False, sort = False)
        for group in groups: #for each of these groups, we need to check for overlap
            image_id = group[0][0] #grabbing id and class_name for later
            class_name = group[0][1]
            group = group[1].drop(["image_id", "class_name"], axis = 1) #drop these cols
            group = group.values.tolist() #getting the actual list of rows (as lists)
            newGroup = []
            while group: #while there are still obs in the group
                obs1 = group.pop() #pop the last guy
                obs1 = torch.tensor([obs1], dtype = torch.float) #turn him into a tensor
                if not newGroup:
                    newGroup.append(obs1)
                else:
                    merged = False #assume no merging happens
                    for obs2 in newGroup:
                        iou = ops.box_iou(obs1, obs2) #check the IOU
                        if iou.numpy()[0][0] >= threshold: #if it's over the threshold set, then merge
                            obs2 = torch.div(torch.add(obs1, obs2), 2) #takes elementwise average of bounding box coordinates
                            merged = True #if we end up merging
                    if merged == False: #if we don't end up merging to anything in newgroup
                        newGroup.append(obs1) #then add it to the newgroup
            for obs in newGroup:
                newAnnos.append([image_id]+[class_name]+obs.tolist()[0])
        newAnnos = pd.DataFrame(newAnnos, columns = ["image_id","class_name","x_min","y_min","x_max","y_max"])
        newAnnos.to_csv(csvName, sep = ",", header=True, index = False)
        newLength = len(newAnnos)
        print(f"{oldLength} -> {newLength} annotations, {oldLength-newLength} annotations merged")
        

#if IOU higher than threshold, then draw bounding box that encompasses both observations
#https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
#https://doi.org/10.1016/j.imavis.2021.104117 #implementation of WBF without the weights from the original paper
def mergeIOUEncompass(annoCSV, threshold = 0.5, csvName = "mergedIOUEncompass.csv"):
    try:
        #read in csv
        oldAnno = pd.read_csv(annoCSV, sep = ",")
    except FileNotFoundError:
        print("CSV not found!")
    else:
        oldLength = len(oldAnno)
        newAnnos = [] #empty list to hold new annotations once they have been created
        if "rad_id" in oldAnno.columns:
            oldAnno = oldAnno.drop("rad_id", axis = 1)
        oldAnno = oldAnno.dropna(axis = 0) #drop all rows with no findings
        oldAnno = oldAnno[["image_id","class_name","x_min","y_min","x_max","y_max"]]
        #groupby both image id and class
        #all the other stuff is for performance
        groups = oldAnno.groupby(["image_id", "class_name"], group_keys = False, as_index = False, sort = False)
        for group in groups: #for each of these groups, we need to check for overlap
            image_id = group[0][0] #grabbing id and class_name for later
            class_name = group[0][1]
            group = group[1].drop(["image_id", "class_name"], axis = 1) #drop these cols
            group = group.values.tolist() #getting the actual list of rows (as lists)
            #print(group)
            newGroup = []
            while group: #while there are still obs in the group
                obs1 = group.pop() #pop the last guy
                obs1 = torch.tensor([obs1], dtype = torch.float) #turn him into a tensor
                if not newGroup:
                    newGroup.append(obs1)
                else:
                    merged = False #assume no merging happens
                    for obs2 in newGroup:
                        #print(obs2)
                        print(obs1, obs2)
                        iou = ops.box_iou(obs1, obs2) #check the IOU
                        if iou.numpy()[0][0] >= threshold: #if it's over the threshold set, then merge
                            minTensor = torch.min(obs1, obs2) #we want x_min and y_min
                            maxTensor = torch.max(obs1, obs2) #we want x_max and y_max
                            obs2 = torch.cat((minTensor[0][0:2], maxTensor[0][2:])) #merge x_min, y_min, x_max and y_max 
                            merged = True #if we end up merging
                    if merged == False: #if we don't end up merging to anything in newgroup
                        newGroup.append(obs1) #then add it to the newgroup
            for obs in newGroup:
                newAnnos.append([image_id]+[class_name]+obs.tolist()[0])
        newAnnos = pd.DataFrame(newAnnos, columns = ["image_id","class_name","x_min","y_min","x_max","y_max"])
        newAnnos.to_csv(csvName, sep = ",", header=True, index = False)
        newLength = len(newAnnos)
        print(f"{oldLength} -> {newLength} annotations, {oldLength-newLength} annotations merged")


#if the bounding boxes exceed the resolution of the image, then reset to boundary (e.g -2.2312 -> 0)
def checkBBs(csvFile, resolution):
    pass

if __name__ == "__main__":
    #mergeIOUAverage("FULL_1024_PAD_annotations/anno_train.csv", 0.3, "mergedAnnoTrainAverage_03.csv")
    #mergeIOUAverage("FULL_1024_PAD_annotations/anno_test.csv", 0.3, "mergedAnnoTestAverage_03.csv")
    
    #mergeIOUEncompass("FULL_1024_PAD_annotations/anno_train.csv", 0.3, "mergedAnnoTrain")
    #mergeIOUEncompass("FULL_1024_PAD_annotations/anno_test.csv", 0.3, "mergedAnnoTestEncompass_0.3.csv")

    
    """box1 = torch.tensor([[0,0,10,0]], dtype = torch.float)
    box2 = torch.tensor([[10,10,0,10]], dtype = torch.float)
    
    print(ops.box_iou(box1, box2))
    
    minTensor = torch.min(box1, box2) #we want x_min and y_min
    maxTensor = torch.max(box1, box2) #we want x_max and y_max
    obs2 = torch.cat((minTensor[0][0:2], maxTensor[0][2:]))
    print(obs2)"""