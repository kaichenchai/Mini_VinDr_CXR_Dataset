#this program converts the CSV files from the original (22 class) observations to the Kaggle observations (14 class)

import os
import pandas as pd

diseasesList = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis"
]

trainDir = "FULL_1024_PAD_annotations/anno_train.csv"
trainCSV = pd.read_csv(trainDir)
testDir = "FULL_1024_PAD_annotations/anno_test.csv"
testCSV = pd.read_csv(testDir)

trainCSV = trainCSV[trainCSV["class_name"].isin(diseasesList)]
testCSV = testCSV[testCSV["class_name"].isin(diseasesList)]

trainCSV.to_csv("kaggleTrain.csv", sep = ",", header=True, index=None)
testCSV.to_csv("kaggleTest.csv", sep = ",", header=True, index=None)