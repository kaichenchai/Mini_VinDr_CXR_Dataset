#This is an attempt to clean up and fix the bounding boxes in the VinDr dataset.
#From my exploration of the dataset, I have found that there are bounding box annotations that are outside
#Of the possible range of the image (negative, greater than x_max and y_max)

import pandas as pd
import numpy as np
import os

#reading in the csvs of image dimensions
try:
    train_dim = pd.read_csv("C:/Users/engli/Documents/Y2S1/NSPC2001/Mini_VinDr_CXR_Dataset/FULL_1024_PAD_annotations/dimensions_train.csv")
    test_dim = pd.read_csv("C:/Users/engli/Documents/Y2S1/NSPC2001/Mini_VinDr_CXR_Dataset/FULL_1024_PAD_annotations/dimensions_test.csv")
except FileNotFoundError as e:
    print(e)
#combining them
combined_dim = pd.concat((train_dim, test_dim))

try:
    anno_train = pd.read_csv("C:/Users/engli/Documents/Y2S1/NSPC2001/Mini_VinDr_CXR_Dataset/original_dataset/annotations/annotations_train.csv")
    anno_test = pd.read_csv("C:/Users/engli/Documents/Y2S1/NSPC2001/Mini_VinDr_CXR_Dataset/original_dataset/annotations/annotations_test.csv")
except FileNotFoundError as e:
    print(e)

combined_anno = pd.concat((anno_train, anno_test))

"""for i, row in combined_anno.iterrows():
    print(row.image_id)
    imgXMax = combined_dim.loc[combined_dim["image_id"] == row.image_id, "x_dim"].values[0]
    print(imgXMax)
    imgYMax = combined_dim.loc[combined_dim["image_id"] == row.image_id, "y_dim"].values[0]
    print(imgYMax)
    #if less than lower bounds of image
    if row.x_min < 0:
        print(f"{row.x_min} -> 0")
        combined_anno.at[i, "x_min"] = 0
    if row.y_min < 0:
        print(f"{row.y_min} -> 0")
        combined_anno.at[i, "y_min"] = 0
    #if greater than upper bounds of image
    if row.x_max > imgXMax:
        print(f"{row.x_max} -> {imgXMax}")
        combined_anno.at[i, "x_max"] = imgXMax
    if row.y_max > imgYMax:
        print(f"{row.y_max} -> {imgYMax}")
        combined_anno.at[i, "y_max"] = imgYMax"""
        
        
combined_anno = pd.merge(combined_anno, combined_dim, on = ["image_id", "image_id"])
combined_anno.loc[combined_anno["x_min"] < 0, "x_min"] = 0
combined_anno.loc[combined_anno["y_min"] < 0, "y_min"] = 0

combined_anno.loc[combined_anno["x_max"] > combined_anno["x_dim"], "x_max"] = combined_anno.loc[combined_anno["x_max"] > combined_anno["x_dim"], "x_dim"]
combined_anno.loc[combined_anno["y_max"] > combined_anno["y_dim"], "y_max"] = combined_anno.loc[combined_anno["y_max"] > combined_anno["y_dim"], "y_dim"]


combined_anno.to_csv("fixedBB.csv", index = None)