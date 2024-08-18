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
        newAnnos = [] #empty list to hold new annotations once they have been created
        if "rad_id" in oldAnno.columns:
            oldAnno = oldAnno.drop("rad_id", axis = 1)
        oldAnno = oldAnno.dropna(axis = 0) #drop all rows with no findings
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
                #print(obs1)
                obs1 = torch.tensor([obs1], dtype = torch.float) #turn him into a tensor
                #print(obs1)
                if not newGroup:
                    #print(obs1.tolist()[0])
                    #newGroup.append(obs1.tolist()[0])
                    newGroup.append(obs1)
                else:
                    merged = False #assume no merging happens
                    for obs2 in newGroup:
                        #print(obs2)
                        #obs2 = torch.tensor([obs2], dtype = torch.float) #turn him into a tensor
                        iou = ops.box_iou(obs1, obs2) #check the IOU
                        if iou.numpy()[0][0] >= threshold: #if it's over the threshold set, then merge
                            obs2 = torch.div(torch.add(obs1, obs2), 2) #takes elementwise average of bounding box coordinates
                            #obs2 = obs2.tolist()[0]
                            merged = True #if we end up merging
                    if merged == False: #if we don't end up merging to anything in newgroup
                        newGroup.append(obs1) #then add it to the newgroup
            for obs in newGroup:
                newAnnos.append([image_id]+[class_name]+obs.tolist()[0])
        newAnnos = pd.DataFrame(newAnnos, columns = ["image_id","class_name","x_min","y_min","x_max","y_max"])
        newAnnos.to_csv(csvName, sep = ",", header=True, index = False)

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
        newAnnos = [] #empty list to hold new annotations once they have been created
        if "rad_id" in oldAnno.columns:
            oldAnno = oldAnno.drop("rad_id", axis = 1)
        oldAnno = oldAnno.dropna(axis = 0) #drop all rows with no findings
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
                #print(obs1)
                obs1 = torch.tensor([obs1], dtype = torch.float) #turn him into a tensor
                #print(obs1)
                if not newGroup:
                    #print(obs1.tolist()[0])
                    #newGroup.append(obs1.tolist()[0])
                    newGroup.append(obs1)
                else:
                    merged = False #assume no merging happens
                    for obs2 in newGroup:
                        #print(obs2)
                        #obs2 = torch.tensor([obs2], dtype = torch.float) #turn him into a tensor
                        iou = ops.box_iou(obs1, obs2) #check the IOU
                        if iou.numpy()[0][0] >= threshold: #if it's over the threshold set, then merge
                            minTensor = torch.min(obs1, obs2) #we want x_min and y_min
                            maxTensor = torch.max(obs1, obs2) #we want x_max and y_max
                            obs2 = torch.cat((minTensor[0][0:2], maxTensor[0][2:])) #takes elementwise average of bounding box coordinates
                            #obs2 = obs2.tolist()[0]
                            merged = True #if we end up merging
                    if merged == False: #if we don't end up merging to anything in newgroup
                        newGroup.append(obs1) #then add it to the newgroup
            for obs in newGroup:
                newAnnos.append([image_id]+[class_name]+obs.tolist()[0])
        newAnnos = pd.DataFrame(newAnnos, columns = ["image_id","class_name","x_min","y_min","x_max","y_max"])
        newAnnos.to_csv(csvName, sep = ",", header=True, index = False)


#if the bounding boxes exceed the resolution of the image, then reset to boundary (e.g -2.2312 -> 0)
def checkBBs(csvFile, resolution):
    

if __name__ == "__main__":
    #mergeIOUAverage("annotationsTrain.csv", 0.5)
    #mergeIOUAverage("annotationsTrain.csv", 0.3)
    
    mergeIOUEncompass("annotationsTrain.csv", 0.3)

    
    """box1 = torch.tensor([[387.86810302734375,254.7804412841797,406.8974304199219,327.2384033203125]], dtype = torch.float)
    box2 = torch.tensor([[393.08123779296875,240.74595642089844,453.1715393066406,333.98931884765625]], dtype = torch.float)
    
    print(ops.box_iou(box1, box2))
    
    print(box1[0][0:2])
    print(box1[0][2:])
    
    print(torch.concat((box1[0][0:2], box1[0][2:])))"""