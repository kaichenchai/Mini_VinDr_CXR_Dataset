import pandas as pd
import numpy as np

#newLabels is a csv file that contains labels and corresponding (0 and 1s) for diseases
#oldCSV is the old CSV file
#newName is the name of the output CSV
def splitTrainingCSV(newLabels, oldCSV, newName):
    try:
        newLabels = pd.read_csv(newLabels)
        oldCSV = pd.read_csv(oldCSV)
    except FileNotFoundError:
        print("File not found at directory")
    else:
        idList = list(newLabels["image_id"])
        print(len(oldCSV))
        newCSV = oldCSV[oldCSV["image_id"].isin(idList)]
        print(len(newCSV))
        newCSV.to_csv(newName, index = None)
        
def combineAndSplitCSV(newLabels: str, oldCSV: list[str], newName: str):
    pass

if __name__ == "__main__":
    splitTrainingCSV("stratifiedSplit/image_labels_trainNEW.csv", "FULL_1024_PAD_annotations/anno_train.csv", "FULL_1024_PAD_annotations/anno_trainNEW.csv")
    splitTrainingCSV("stratifiedSplit/image_labels_valNEW.csv", "FULL_1024_PAD_annotations/anno_train.csv", "FULL_1024_PAD_annotations/anno_valNEW.csv")
    splitTrainingCSV("stratifiedSplit/image_labels_trainNEW.csv", "FULL_1024_PAD_annotations/kaggleTrain.csv", "FULL_1024_PAD_annotations/kaggleTrainNEW.csv")
    splitTrainingCSV("stratifiedSplit/image_labels_valNEW.csv", "FULL_1024_PAD_annotations/kaggleTrain.csv", "FULL_1024_PAD_annotations/kaggleValNEW.csv")