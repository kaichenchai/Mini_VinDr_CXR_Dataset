import argparse
from pathlib import Path
import os

from imageModifications import resizeWithPadding

from ultralytics import YOLO
import pandas as pd
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, help="Path to folder")
    parser.add_argument("--base_path", type=str, help="The base path for images")
    parser.add_argument("--results_path", type=str, help="Path for results csv")
    parser.add_argument("--weights_path", type=str, help="For loading into ultralytics")
    args = parser.parse_args()
    
    model = YOLO(args.weights_path)
    
    input_csv = pd.read_csv(args.csv_path)
    input_csv["full_path"] = input_csv["full_path"].apply(lambda x: os.path.join(args.base_path, x))
    images_dir = input_csv["full_path"].to_list
    image_ids = [Path(dir).stem for dir in images_dir]
    
    dfs = []
    image_id_list = []
    for dir, id in zip(images_dir, image_ids):
        data = cv2.imread(dir)
        data = cv2.equalizeHist(data)
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
            image_id_list.extend([id]*len(result))
            if args.show_pred:
                result.show()
            

    df = pd.concat(dfs, ignore_index=True)
    df["image_id"] = image_id_list
    if args.results_path:
        df.to_csv(args.results_path, index=None)