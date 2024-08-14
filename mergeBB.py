import numpy as np
from PIL import Image, ImageOps
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import ops


#https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
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
        imageIDList = oldAnno["image_id"].unique() #gets unique image_id -> should be 15,000
        """for ID in imageIDList:
            subset = oldAnno.loc[oldAnno["image_id"] == ID] #gets subset of annotations based on image_id
            #for each disease, we need to check the instances to see if they overlap
            conditions = subset["class_name"].unique() 
            for condition in conditions: #splits subset by disease
                conditionSubset = subset.loc[subset["class_name"] == condition]
                conditionSubset = conditionSubset.drop(labels = ["image_id", "class_name"], axis = 1)
                conditionSubset = conditionSubset.values.tolist()
                for condition in conditionSubset"""
        #groupby both image id and class
        #all the other stuff is for performance
        groups = oldAnno.groupby(["image_id", "class_name"], group_keys = False, as_index = False, sort = False)
        for group in groups: #for each of these groups, we need to check for overlap
            id = group[0][0] #grabbing id and class_name for later
            class_name = group[0][1]
            group = group[1].drop(["image_id", "class_name"], axis = 1) #drop these cols
            group = group.values.tolist() #getting the actual list of rows (as lists)
            while group: #while there are still obs in the group
                obs1 = group.pop() #pop the last guy
                obs1 = torch.tensor([obs1], dtype = torch.float) #turn him into a tensor
                for i in range(len(group)-1): #for every other observation
                    obs2 = torch.tensor([group.pop()], dtype = torch.float) #turn him into a tensor
                    iou = ops.box_iou(obs1, obs2) #check the IOU
                    if iou.numpy()[0][0] > threshold: #if it's over the threshold set, then merge
                        newcoords = torch.div(torch.add(obs1, obs2), 2) #takes average of bounding box coordinates
                        group.append(newcoords.tolist()[0])
                        print(f"APPENDED | 1: {obs1}, 2: {obs2}, iou: {iou}, new: {newcoords}")
                    else:

#currently not working
                        

if __name__ == "__main__":
    mergeIOUAverage("annotationsTrain.csv", 0.5)