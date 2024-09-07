import pandas as pd
import os
import shutil

#function that moves images within a csv to a new location
#the image names should be within a column called image_id and not contain the filename
#assumes PNG
def moveFiles(csv, oldDir = None, newDir = None):
    try:
        csv = pd.read_csv(csv)
    except FileNotFoundError:
        print("File not found in directory")
    idList = list(set(csv["image_id"])) #gets unique image_ids, order doesn't matter
    os.makedirs(newDir, exist_ok=True)
    for id in idList:
        fileName = f"{id}.png"
        src = os.path.join(oldDir, fileName)
        dest = os.path.join(newDir, fileName)
        if os.path.exists(src):
            shutil.move(src, dest)
            print(f"Moved: {fileName}")
        else:
            print(f"File not found: {fileName}")

if __name__ == "__main__":
    moveFiles("FULL_1024_PAD_annotations/anno_trainNEW.csv")