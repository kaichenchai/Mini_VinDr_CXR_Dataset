from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os
"""from wandb.integration.ultralytics import add_wandb_callback
import wandb"""


"""wandb.init(project="VinDr_YOLOv8", job_type="inference", name = "Inference_Testing_Specific_Classes",
config={
    "dataset": "SUBSET_brightnessEQ",
    "model": "YOLOv8n",
    "image_size": 1280,
    "machine": "Thermaltake_2080ti_0"
}
)"""

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

#Importing a csv with 0/1 for classes within the image
dirToObs = "C:/Users/engli/Documents/Y2S1/NSPC2001/Mini_VinDr_CXR_Dataset/original_dataset/annotations/mini_image_labels_test.csv"
classes = pd.read_csv(dirToObs)
classes = classes.drop(["COPD","Lung tumor","Pneumonia","Tuberculosis","Other disease","No finding"], axis = 1)

#reading in the ground truth for images
dirToGT = "C:/Users/engli/Documents/Y2S1/NSPC2001/Mini_VinDr_CXR_Dataset/original_dataset/annotations/annotations_test.csv"
groundTruth = pd.read_csv(dirToGT)

#Reading in the model to YOLO
dirToModel = "YOLOv8/brightnessEQFIXED_best.pt"
model = YOLO(dirToModel)

#Setting directory of images
imageDir = "C:/Users/engli/Documents/Y2S1/NSPC2001/Mini_VinDr_CXR_Dataset/1024_brightnessEQ_dataset/images/val/"
images = os.listdir(imageDir)

#makes a new row called "new" that contains a list of all the conditions that match 1
classes['conditions'] = classes.apply(lambda x: x.index[x == 1].tolist(), axis=1)
classes.set_index("image_id", inplace = True)

#to hold the results after inference
resultsList = []

for imageName in images:
    imageID = imageName[:-4]
    conditions = classes.loc[imageID]["conditions"]
    if conditions:
        indexes = [classesDict[x] for x in conditions] #get the indexes of each disease
        resultsList.append(model.predict(imageDir + imageName, classes = indexes, conf = 0.25, imgsz = 1024, show = True))
        
    