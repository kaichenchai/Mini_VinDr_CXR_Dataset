import pydicom
from PIL import Image
import os
import numpy as np
import pandas as pd

#directory to images to be converted

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
            dicom = pydicom.dcmread(fullpath)
            image = dicom.pixel_array
            img_8bit = ((image / image.max()) * 255).astype('uint8')
            img_pil = Image.fromarray(img_8bit)
            img_pil.save(os.path.join(outputDir, filename.replace('.dicom', '.png')))
    print("Finished!!")

#Gets dimensions of .png fiiles from a directory and returns it as a dictionary
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
def makeLabels(labelsDir, dimDict = None, classDict = None):
    fileDf = pd.read_csv(labelsDir)
    fileDf['class_name'] = fileDf['class_name'].map(classDict)

if __name__ == "__main__":
    #convertDicom("dataset/train_subset", "images/train") #as already been up
    #dimDict = getDim("dataset/train_subset")
    dimDict = getDim("images/train")
    #makeLabels("labels/train", dimDict, classesDict)