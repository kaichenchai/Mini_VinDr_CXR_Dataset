import os
import pandas as pd

def mergeObs(csvDir, outputName = "mergedOutput.csv"):
    try:
        df = pd.read_csv(csvDir)
        if df is None:
            raise FileNotFoundError
    except FileNotFoundError as e:
        print("File not found!")
        
    df = df.dropna(axis = 1)
    groups = df.groupby(["image_id", "class_name"], group_keys = False, as_index = False, sort = False)
    for key, obs in groups:
        if len(obs) > 1:
            print(key, obs)
            
if __name__ == "__main__":
    mergeObs("../FULL_1024_PAD_annotations/anno_train.csv", "mergedTrain.csv")