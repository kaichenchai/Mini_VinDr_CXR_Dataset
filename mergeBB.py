import numpy as np
from PIL import Image, ImageOps
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import ops

def mergeIOUAverage(annoCSV, threshold = 0.5, csvName = "mergedIOUAverage.csv"):
    try:
        #read in csv
        oldAnno = pd.read_csv(annoCSV, sep = ",")
    except FileNotFoundError:
        print("CSV not found!")
    else:
        oldAnno.dropna(axis = 0) #drop all rows with no findings
        imageIDList = oldAnno["image_id"].unique() #gets unique image_id
        for ID in imageIDList:
            subset = oldAnno.loc[oldAnno["image_id"] == ID] #gets subset of annotations based on image_id
        
if __name__ == "__main__":
    mergeIOUAverage("annotationsTrain.csv", 0.5)