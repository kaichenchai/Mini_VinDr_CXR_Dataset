import pandas as pd
import os
import shutil

#function that moves images within a csv to a new location
#the image names should be within a column called image_id and not contain the filename
#assumes PNG
def moveFiles(csv: str, oldDir: str, newDir: str):
    try:
        csv = pd.read_csv(csv)
    except FileNotFoundError:
        print("File not found in directory")
    idList = list(set(csv["image_id"])) #gets unique image_ids, order doesn't matter
    os.makedirs(newDir, exist_ok=True)
    notMoved = 0
    for id in idList:
        fileName = f"{id}.png"
        src = os.path.join(oldDir, fileName)
        dest = os.path.join(newDir, fileName)
        if os.path.exists(src):
            shutil.move(src, dest)
            print(f"Moved: {fileName}")
        else:
            print(f"File not found: {fileName}")
            notMoved += 1
    print(f"Started with {len(idList)}, didn't manage to move {notMoved}")
    
#checks folder against csv to see if there are any files that should not be there
#also says if there are files that should be there, but are not
def checkFolder(csv: str, dir: str):
    try:
        csv = pd.read_csv(csv)
    except FileNotFoundError:
        print("File not found in directory")
    idList = list(set(csv["image_id"])) #gets unique image_ids, order doesn't matter
    filesInDir = os.listdir(dir)
    for i in range(len(filesInDir)):
        filesInDir[i] = filesInDir[i][:-4]
    print(f"Files not in folder {[x for x in idList if x not in filesInDir]}")
    print(f"Files that should not be in folder {[x for x in filesInDir if x not in idList]}")

if __name__ == "__main__":
    #moveFiles("/mnt/data/kai/VinDr_Code/Mini_VinDr_CXR_Dataset/subset/c-subset/explainability/cardiomegaly_subset_to_annotate.csv", "/mnt/data/kai/VinDr_Subsets/YOLOv8_format_datasets/FULL_1024_brightnessEQ_FIXED/images/train1", "/mnt/data/kai/VinDr_Subsets/cardiomegaly_subset/1024_padding_brightnessEQ/train")
    #moveFiles("stratifiedSplit/image_labels_valNEW.csv", "1024_brightnessEQ_dataset/images/train", "1024_brightnessEQ_dataset/images/val")
    checkFolder("/mnt/data/kai/VinDr_Code/Mini_VinDr_CXR_Dataset/subset/c-subset/explainability/cardiomegaly_subset_to_annotate.csv", "/mnt/data/kai/VinDr_Subsets/cardiomegaly_subset/1024_padding_brightnessEQ/train")
    #checkFolder("stratifiedSplit/image_labels_trainNEW.csv", "1024_brightnessEQ_dataset/images/train")
