import numpy as np
from PIL import Image, ImageOps
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import ops

def mergeIOUAverage(annoCSV, threshold, csvName = "mergedIOUAverage.csv"):
    try:
        #read in csv
        oldAnno = pd.read_csv(annoCSV, sep = ",")
    except FileNotFoundError:
        print("CSV not found!")
    else:
        #drop all rows with no findings
        oldAnno.dropna(axis = 0)