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

try:
    train_anno = pd.read_csv("C:/Users/engli/Documents/Y2S1/NSPC2001/Mini_VinDr_CXR_Dataset/original_dataset/annotations/annotations_train.csv")
    test_anno = pd.read_csv("C:/Users/engli/Documents/Y2S1/NSPC2001/Mini_VinDr_CXR_Dataset/original_dataset/annotations/annotations_test.csv")
except FileNotFoundError as e:
    print(e)


def fixBB(annoDf, dimDf, csvName):
    combined_anno = pd.merge(annoDf, dimDf, on = ["image_id", "image_id"])
    combined_anno.loc[combined_anno["x_min"] < 0, "x_min"] = 0
    combined_anno.loc[combined_anno["y_min"] < 0, "y_min"] = 0
    
    combined_anno.loc[combined_anno["x_max"] > combined_anno["x_dim"], "x_max"] = combined_anno.loc[combined_anno["x_max"] > combined_anno["x_dim"], "x_dim"]
    combined_anno.loc[combined_anno["y_max"] > combined_anno["y_dim"], "y_max"] = combined_anno.loc[combined_anno["y_max"] > combined_anno["y_dim"], "y_dim"]

    #these ones will be out of range by default, just drop these rows
    combined_anno = combined_anno.drop(combined_anno[combined_anno["x_min"] >= combined_anno["x_dim"]].index)
    combined_anno = combined_anno.drop(combined_anno[combined_anno["y_min"] >= combined_anno["y_dim"]].index)
    combined_anno = combined_anno.drop(combined_anno[combined_anno["x_max"] <= 0].index)
    combined_anno = combined_anno.drop(combined_anno[combined_anno["y_max"] <= 0].index)
    
    combined_anno.to_csv(csvName, index = None)


fixBB(train_anno, train_dim, "fixedBBtrain.csv")
fixBB(test_anno, test_dim, "fixedBBtest.csv")