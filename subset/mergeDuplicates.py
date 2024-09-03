import os
import pandas as pd

#for aortic enlargement and cardiomegaly, we know that for each image,
#there should be max 1 instance of both diseases
#hence merge all instances by taking an average 
def mergeObs(csvDir, outputName = "mergedOutput.csv"):
    try:
        df = pd.read_csv(csvDir) #read in csv
        if df is None:
            raise FileNotFoundError
    except FileNotFoundError as e:
        print("File not found!")
        
    df = df.dropna(axis = 1) #drop all empty rows
    groups = df.groupby(["image_id", "class_name"], group_keys = False, as_index = False, sort = False)
    newAnnos = []
    for key, obs in groups:
        if len(obs) > 1:
            xMinAvg = obs["x_min"].mean()
            xMaxAvg = obs["x_max"].mean()
            yMinAvg = obs["y_min"].mean()
            yMaxAvg = obs["y_max"].mean()
            line = [key[0][0]]+[key[0][1]]+[xMinAvg, yMinAvg, xMaxAvg, yMaxAvg]
            print(line)
            newAnnos.append(line)
    newAnnos.to_csv(outputName, sep = ",", header = True, index = False)
            
if __name__ == "__main__":
    mergeObs("train.csv", "mergedTrain.csv")