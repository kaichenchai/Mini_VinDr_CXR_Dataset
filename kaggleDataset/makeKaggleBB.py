import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import os
import numpy as np
import pandas as pd

classesDict = {
    "Aortic enlargement":0,
    "Atelectasis":1,
    "Calcification":2,
    "Cardiomegaly":3,
    "Consolidation":4,
    "ILD":5,
    "Infiltration":6,
    "Lung Opacity":7,
    "Nodule/Mass":8,
    "Other lesion":9,
    "Pleural effusion":10,
    "Pleural thickening":11,
    "Pneumothorax":12,
    "Pulmonary fibrosis":13
}

#making labels from set dimensions
def makeLabelsSetDim(labelsDir, outputDir, dim, classDict):
    fileDf = pd.read_csv(labelsDir)
    fileDf['class_name'] = fileDf['class_name'].map(classDict)
    x_dim = dim[0]
    y_dim = dim[1]
    for row in fileDf.itertuples():
        scanID = row[1]
        classNum = row.class_name
        x_min = row.x_min
        y_max = row.y_max #annotations also count from top left corner to bottom left corner
        x_max = row.x_max
        y_min = row.y_min
        
        #converting into x_center, y_center, x_width, y_height
        x_center = (x_min+x_max)/2
        y_center = (y_min+y_max)/2
        x_width = x_max-x_min
        y_height = y_max-y_min
        
        #normalising to proportion of overall size
        x_center_norm = x_center/x_dim
        y_center_norm = y_center/y_dim
        x_width_norm = x_width/x_dim
        y_height_norm = y_height/y_dim
        
        txtFilename = scanID + ".txt"
        txtFilePath = os.path.join(outputDir, txtFilename)
        with open(txtFilePath, "a") as file:
            file.write(f"{classNum} {x_center_norm} {y_center_norm} {x_width_norm} {y_height_norm}\n")
            
if __name__ == "__main__":
        makeLabelsSetDim("FULL_1024_PAD_annotations/kaggleTrain.csv", "1024_brightnessEQ_dataset/labels/train", (1024, 1024), classesDict)