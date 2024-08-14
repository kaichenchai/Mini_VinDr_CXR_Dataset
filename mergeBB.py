import numpy as np
from PIL import Image, ImageOps
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import ops


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
            id = group[0][0] #grabbing id and class_name for later
            class_name = group[0][1]
            group = group[1].drop(["image_id", "class_name"], axis = 1) #drop these cols
            group = group.values.tolist() #getting the actual list of rows (as lists)
            print(group)
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
                        if iou.numpy()[0][0] > threshold: #if it's over the threshold set, then merge
                            obs2 = torch.div(torch.add(obs1, obs2), 2) #takes average of bounding box coordinates
                            obs2 = obs2.tolist()[0]
                            merged = True #if we end up merging
                    if merged == False: #if we don't end up merging
                        #newGroup.append(obs1.tolist()[0])
                        newGroup.append(obs1) #then add it to the newgroup
            print(newGroup)
                        

if __name__ == "__main__":
    mergeIOUAverage("annotationsTrain.csv", 0.5)