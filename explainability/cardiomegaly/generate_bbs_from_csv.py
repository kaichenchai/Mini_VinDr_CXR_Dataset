import argparse
from pathlib import Path
import os

from ultralytics import YOLO
import pandas as pd
import cv2

def resizeWithPadding(image, newDim):
    #image: np.array
    #newDim: tuple of (newx, newy) dimensions

    oldDim = (image.shape[0], image.shape[1]) #gets the dimensions of original img (y, x)
    ratio = float(max(newDim)/max(oldDim)) #finds the tightest ratio
    convertedDim = tuple([int(x*ratio) for x in oldDim]) #makes a tuple of dim
    newImg = cv2.resize(image, (convertedDim[1], convertedDim[0])) #converts image to size, need to swap to (x, y)

    delta_w = newDim[1] - convertedDim[1] #as newDim and convertedDim in (y, x) format
    delta_h = newDim[0] - convertedDim[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0] #fills with black
    newImg = cv2.copyMakeBorder(newImg, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return newImg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to folder")
    parser.add_argument("--base_path", type=str, required=True, help="The base path for images")
    parser.add_argument("--results_path", type=str, required=True, help="Path for results csv")
    parser.add_argument("--weights_path", type=str, required=True, help="For loading into ultralytics")
    args = parser.parse_args()
    
    model = YOLO(args.weights_path)
    
    input_csv = pd.read_csv(args.csv_path)
    input_csv["full_path"] = input_csv["Frontal_Image_Path"].apply(lambda x: os.path.join(args.base_path, x))
    images_dir = input_csv["full_path"].to_list()
    image_ids = [Path(dir).stem for dir in images_dir]

    dfs = []
    image_id_list = []
    for full_path, dir in zip(images_dir, input_csv["Frontal_Image_Path"].to_list()):
        data = cv2.imread(full_path)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        data = cv2.equalizeHist(data)
        data = cv2.cvtColor(data,cv2.COLOR_GRAY2BGR)
        data = resizeWithPadding(data, (1024,1024))
        
        results = model.predict(
            source = data,
            imgsz = 1024,
            conf = 0.40,
            iou = 0.6,
            max_det = 2
        )
        
        for result in results:
            dfs.append(result.to_df())
            image_id_list.extend([dir]*len(result))

    df = pd.concat(dfs, ignore_index=True)
    df["image_id"] = image_id_list
    if args.results_path:
        df.to_csv(args.results_path, index=None)
