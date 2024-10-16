import pydicom
from PIL import Image
import os
import numpy as np

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

def convertDicomAndBB(inputDir, outputDir):
    fileDict = {}
    filenames = os.listdir(inputDir)
    for filename in filenames:
        if filename.endswith(".dicom"):
            fullpath = os.path.join(inputDir, filename)
            dicom = pydicom.dcmread(fullpath)
            image = dicom.pixel_array
            img_8bit = ((image / image.max()) * 255).astype('uint8')
            img_pil = Image.fromarray(img_8bit)
            img_pil.save(os.path.join(outputDir, filename.replace('.dicom', '.png')))
            filenameStripped = filename.with_suffix("")
            dim = image.shape
            fileDict[filenameStripped] = dim
    print(fileDict)
            

if __name__ == "__main__":
    convertDicomAndBB("dataset/test_subset", "images/train")