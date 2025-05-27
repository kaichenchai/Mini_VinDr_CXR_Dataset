import numpy as np
import pydicom
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

"""img = cv2.imread("/Users/kaichenchai/Documents/Y2S1/NPSC2001/images/train/0b5d222662dfa80d7ba8b101bb88e20c.png")
cv2.imshow("hello world", img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

imgDir = "./image_aura_demo/"
print(imgDir)

#reading in csv of annotations
annoPath = "image_aura_demo/chayan_cardiomegaly.csv"
annotations = pd.read_csv(annoPath, sep=",")
print(annotations.shape)

#bounding box and text colour dictionary
color_dict = {
    "Aortic enlargement": (255, 0, 0),    # Red
    "Atelectasis": (0, 255, 0),    # Green
    "Calcification": (0, 0, 255),    # Blue
    "Cardiomegaly": (255, 255, 0),  # Yellow
    "Clavicle fracture": (255, 0, 255),  # Magenta
    "Consolidation": (0, 255, 255),  # Cyan
    "Edema": (128, 0, 0),    # Maroon
    "Emphysema": (0, 128, 0),    # Dark Green
    "Enlarged PA": (0, 0, 128),    # Navy
    "ILD": (128, 128, 0), # Olive
    "Infiltration": (128, 0, 128), # Purple
    "Lung Opacity": (0, 128, 128), # Teal
    "Lung cavity": (192, 192, 192), # Silver
    "Lung cyst": (128, 128, 128), # Gray
    "Mediastinal shift": (255, 165, 0),   # Orange
    "Nodule/Mass": (255, 20, 147),  # Deep Pink
    "Pleural effusion": (75, 0, 130),    # Indigo
    "Pleural thickening": (173, 255, 47),  # Green Yellow
    "Pneumothorax": (255, 105, 180), # Hot Pink
    "Pulmonary fibrosis": (0, 191, 255),   # Deep Sky Blue
    "Rib fracture": (139, 69, 19),   # Saddle Brown
    "Other lesion": (255, 228, 181)  # Moccasin
}


choice = None

while choice != "x":
    choice = input("Image filename (x to close): ").lower()
    match choice:
        case "x":
            print("Closing")
        case _:
            try:
                # if choice[-4:] != ".png":
                    # choice = choice + ".png"
                dir = os.path.join(imgDir, choice) #getting the directory of image
                print(dir)
                data = cv2.imread(dir) #reading in the image
                if data is None:
                    raise FileNotFoundError
            except FileNotFoundError:
                print("Image not found in folder")
            else:
                print("Read in image!")
                imgAnno = annotations.loc[annotations["image_id"] == choice[:-4]] #getting all annotations that match image_id
                imgAnno = imgAnno.dropna(axis = 0)
                for row in imgAnno.itertuples(index = False):
                    #drawing rectanges
                    print(row)
                    color = color_dict.get(str(row.class_name), (0, 255, 0))
                    cv2.rectangle(img = data, pt1 = (int(row.x_min), int(row.y_min)), pt2 = (int(row.x_max), int(row.y_max)), color = color, thickness = 2)
                    text = str(row.class_name)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 2
                    cv2.putText(data, text, (int(row.x_min), int(row.y_min)-10), font, font_scale, color, thickness)
                cv2.imshow(choice, data)
                cv2.imwrite("temp.png", data)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                

if __name__ == "main":
    pass