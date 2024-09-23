import os
import pandas as pd

#for aortic enlargement and cardiomegaly, we know that for each image,
#there should be max 1 instance of both diseases
#hence merge all instances by taking an average 
def mergeObsMean(csvDir, outputName = "mergedOutput.csv"):
    try:
        df = pd.read_csv(csvDir) #read in csv
        if df is None:
            raise FileNotFoundError
    except FileNotFoundError as e:
        print("File not found!")
        
    df = df.dropna(axis = 0) #drop all empty rows
    initialNumRows = df.shape[0]
    print(f"Inital number of observations: {initialNumRows}")
    groups = df.groupby(["image_id", "class_name"], group_keys = False, as_index = False, sort = False)
    newAnnos = []
    for key, obs in groups: #for all groups, if 1 obs, then getting mean of it will equal the same value anyway
        xMinAvg = obs["x_min"].mean() #get mean of all values
        xMaxAvg = obs["x_max"].mean()
        yMinAvg = obs["y_min"].mean()
        yMaxAvg = obs["y_max"].mean()
        line = [key[0], key[1], xMinAvg, yMinAvg, xMaxAvg, yMaxAvg] #making a new line of observations
        newAnnos.append(line) #appending line to list of new annotations
    newAnnos = pd.DataFrame(newAnnos, columns = ["image_id","class_name","x_min","y_min","x_max","y_max"]) #converting to dataframe
    newAnnos.to_csv(outputName, sep = ",", header = True, index = False)
    numRowsRemoved = initialNumRows - df.shape[0]
    print(f"Number of duplicate observations removed: {initialNumRows}")
    
def mergeObsEncompass(csvDir, outputName = "mergedOutput.csv"):
    try:
        df = pd.read_csv(csvDir) #read in csv
        if df is None:
            raise FileNotFoundError
    except FileNotFoundError as e:
        print("File not found!")
        
    df = df.dropna(axis = 0) #drop all empty rows
    initialNumRows = df.shape[0]
    print(f"Inital number of observations: {initialNumRows}")
    groups = df.groupby(["image_id", "class_name"], group_keys = False, as_index = False, sort = False)
    newAnnos = []
    for key, obs in groups: #for all groups, if 1 obs, then getting mean of it will equal the same value anyway
        xMinAvg = obs["x_min"].min() #get mean of all values
        xMaxAvg = obs["x_max"].max()
        yMinAvg = obs["y_min"].min()
        yMaxAvg = obs["y_max"].max()
        line = [key[0], key[1], xMinAvg, yMinAvg, xMaxAvg, yMaxAvg] #making a new line of observations
        newAnnos.append(line) #appending line to list of new annotations
    newAnnos = pd.DataFrame(newAnnos, columns = ["image_id","class_name","x_min","y_min","x_max","y_max"]) #converting to dataframe
    newAnnos.to_csv(outputName, sep = ",", header = True, index = False)
    numRowsRemoved = initialNumRows - newAnnos.shape[0]
    print(f"Number of duplicate observations removed: {numRowsRemoved}")
            
if __name__ == "__main__":
    #mergeObsEncompass("train.csv", "mergedTrain.csv")
    #mergeObsEncompass("val.csv", "mergedVal.csv")
    #mergeObsEncompass("test.csv", "mergedTest.csv")
    mergeObsEncompass("ao-subset/train-AO.csv", "mergedTrain-AO.csv")
    mergeObsEncompass("ao-subset/val-AO.csv", "mergedVal-AO.csv")
    mergeObsEncompass("ao-subset/test-AO.csv", "mergedTest-AO.csv")