import pandas as pd
import os

#function that moves images within a csv to a new location
#the image names should be within a column called image_id and not contain the filename
#assumes PNG
def moveFiles(csv, oldDir = None, newDir = None):
    try:
        csv = pd.read_csv(csv)
    except FileNotFoundError:
        print("File not found in directory")
    idList = list(set(csv["image_id"])) #gets unique image_ids, order doesn't matter
    print(idList)
    print(len(idList))

if __name__ == "__main__":
    moveFiles("FULL_1024_PAD_annotations/anno_trainNEW.csv")