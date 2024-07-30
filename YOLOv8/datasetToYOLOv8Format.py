import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import os
import numpy as np
import pandas as pd

#classes and corresponding labels for dataset
classesDict = {'Aortic enlargement': 0,
'Atelectasis': 1,
'Calcification': 2,
'Cardiomegaly': 3,
'Clavicle fracture': 4,
'Consolidation': 5,
'Edema': 6,
'Emphysema': 7,
'Enlarged PA': 8,
'ILD': 9,
'Infiltration': 10,
'Lung Opacity': 11,
'Lung cavity': 12,
'Lung cyst': 13,
'Mediastinal shift': 14,
'Nodule/Mass': 15,
'Pleural effusion': 16,
'Pleural thickening': 17,
'Pneumothorax': 18,
'Pulmonary fibrosis': 19,
'Rib fracture': 20,
'Other lesion': 21,
'COPD': 22,
'Lung tumor': 23,
'Pneumonia': 24,
'Tuberculosis': 25,
'Other disease': 26,
'No finding': 27}

#converts images from dicom format to png
def convertDicom(inputDir, outputDir):
    filenames = os.listdir(inputDir)    
    for filename in filenames:
        if filename.endswith(".dicom"):
            fullpath = os.path.join(inputDir, filename)
            dicom = pydicom.read_file(fullpath)
            data = apply_voi_lut(dicom.pixel_array, dicom)
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                data = np.amax(data) - data
            data = data - np.min(data)
            data = data / np.max(data)
            data = (data * 255).astype(np.uint8)
            img_pil = Image.fromarray(data)
            img_pil.save(os.path.join(outputDir, filename.replace('.dicom', '.png')))
    print("Finished!!")

#Gets dimensions of .png files from a directory and returns it as a dictionary
def getDim(inputDir):
    fileDict = {}
    filenames = os.listdir(inputDir)    
    for filename in filenames:
        if filename.endswith(".png"):
            fullpath = os.path.join(inputDir, filename)
            image = Image.open(fullpath)
            filenameStripped = os.path.splitext(filename)[0]
            dim = image.size
            fileDict[filenameStripped] = dim
    return fileDict

#Need to make labels by grouping based on 
def makeLabels(labelsDir, outputDir, dimDict, classDict):
    fileDf = pd.read_csv(labelsDir)
    fileDf['class_name'] = fileDf['class_name'].map(classDict)
    for row in fileDf.itertuples():
        scanID = row[1] 
        #need to put inside of try except block because annotations_train.csv is full dataset annotations
        try:
            x_dim = dimDict[scanID][0]
            y_dim = dimDict[scanID][1]
        except KeyError as e:
            print(f"{e} not in the dictionary")
        else:
            classNum = row[3]
            x_min = row[4]
            y_max = row[5] #in this case, we swap y_min and y_max as yolov8 counts the y axis from the top of the image
            x_max = row[6]
            y_min = row[7]
            
            #converting into x_center, y_center, x_width, y_height
            x_center = (x_min+x_max)/2
            y_center = (y_min+x_max)/2
            x_width = x_max-x_min
            y_height = y_min-y_max
            
            #normalising to proportion of overall size
            x_center_norm = x_center/x_dim
            y_center_norm = y_center/y_dim
            x_width_norm = x_width/x_dim
            y_height_norm = y_height/y_dim
            
            if classNum != 27: #as do not need a .txt annotation for for no findings
                txtFilename = scanID + ".txt"
                txtFilePath = os.path.join(outputDir, txtFilename)
                with open(txtFilePath, "a") as file:
                    file.write(f"{classNum} {x_center_norm} {y_center_norm} {x_width_norm} {y_height_norm}\n")
                    
#as validation dataset is missing a column                
def makeLabelsVal(labelsDir, outputDir, dimDict, classDict):
    fileDf = pd.read_csv(labelsDir)
    fileDf['class_name'] = fileDf['class_name'].map(classDict)
    for row in fileDf.itertuples():
        scanID = row[1] 
        #need to put inside of try except block because annotations_train.csv is full dataset annotations
        try:
            x_dim = dimDict[scanID][0]
            y_dim = dimDict[scanID][1]
        except KeyError as e:
            print(f"{e} not in the dictionary")
        else:
            classNum = row[2]
            x_min = row[3]
            y_max = row[4] #in this case, we swap y_min and y_max as yolov8 counts the y axis from the top of the image
            x_max = row[5]
            y_min = row[6]
            
            #converting into x_center, y_center, x_width, y_height
            x_center = (x_min+x_max)/2
            y_center = (y_min+x_max)/2
            x_width = x_max-x_min
            y_height = y_min-y_max
            
            #normalising to proportion of overall size
            x_center_norm = x_center/x_dim
            y_center_norm = y_center/y_dim
            x_width_norm = x_width/x_dim
            y_height_norm = y_height/y_dim
            
            if classNum != 27: #as do not need a .txt annotation for for no findings
                txtFilename = scanID + ".txt"
                txtFilePath = os.path.join(outputDir, txtFilename)
                with open(txtFilePath, "a") as file:
                    file.write(f"{classNum} {x_center_norm} {y_center_norm} {x_width_norm} {y_height_norm}\n")

if __name__ == "__main__":
    convertDicom("original_dataset/train_subset", "new_dataset/images/train")
    dimDict = getDim("new_dataset/images/train")
    makeLabels("original_dataset/annotations/annotations_train.csv", "new_dataset/labels/train", dimDict, classesDict) #as done
    
    convertDicom("original_dataset/test_subset", "new_dataset/images/val")
    valDimDict = getDim("new_dataset/images/val")
    makeLabelsVal("original_dataset/annotations/annotations_test.csv", "new_dataset/labels/val", valDimDict, classesDict)
    pass