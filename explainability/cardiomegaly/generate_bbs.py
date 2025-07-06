import glob
import argparse
from pathlib import Path
from traitlets import Bool
from ultralytics import YOLO
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, help="Path to folder")
    #parser.add_argument("--image_output_path", type=str, help="Output path for annotated results")
    parser.add_argument("--results_path", type=str, help="Path for results csv")
    parser.add_argument("--show_pred", type=Bool, help="whether to show results")
    args = parser.parse_args()
    model = YOLO("/Users/kaichenchai/Documents/Projects/Mini_VinDr_CXR_Dataset/explainability/cardiomegaly/weights/YOLOv11m_last.pt")
    
    images_dir = [images for images in glob.iglob(f"{args.images_path}/*.png")]
    image_ids = [Path(dir).stem for dir in images_dir]
    
    dfs = []
    image_id_list = []
    for dir, id in zip(images_dir, image_ids):    
        results = model.predict(
            source = dir,
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