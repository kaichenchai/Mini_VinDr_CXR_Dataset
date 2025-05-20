from pybboxes import BoundingBox
import numpy as np
import pandas as pd


input_file = "FULL_1024_PAD_annotations/15-3-3_split/anno_trainNEW.csv"
csv = pd.read_csv(input_file)
new_file_list = []
for index, row in csv.iterrows():
    box = row[["x_min", "y_min", "x_max", "y_max"]].tolist()
    voc_bbox = BoundingBox.from_voc(*box, image_size = (1024, 1024))
    coco_bbox = voc_bbox.to_coco(return_values = True)
    print(coco_bbox)