import os
import pandas as pd

#removes observations not in a dictionary of observations
def removeObs(csvDir, obsDict, outputName = "output.csv"):
    try:
        csv = pd.read_csv(csvDir)
        if csv is None:
            raise FileNotFoundError
    except FileNotFoundError as e:
        print("File not found!")
        
    csv = csv[csv["class_name"].isin(list(obsDict.keys()))]
    csv.to_csv(outputName, sep = ",", header = True, index = None)

if __name__ == "__main__":
    classesDict = {
    "Aortic enlargement":0,
    "Cardiomegaly":1,
    }
    
    removeObs("FULL_1024_PAD_annotations/anno_train.csv", classesDict, "train.csv")
    removeObs("FULL_1024_PAD_annotations/anno_test.csv", classesDict, "test.csv")