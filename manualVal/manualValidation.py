from ultralytics import YOLO
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score

modelDir = "best.pt"
sourceImgDir = "./images"
saveDir = "./output"

resultsList = []

try:
    images = os.listdir(sourceImgDir)
except FileNotFoundError as e:
    print(e)

model = YOLO(modelDir)

for image in images:
    imgDir = os.path.join(sourceImgDir, image)
    results = model.predict(
        source = imgDir,
        imgsz = 1024,
        conf = 0.0001,
        iou = 0.6,
        max_det = 1,
    )
    image_id = [image[:-4]]
    for r in results:
        conf = r.boxes.conf.tolist()
        box = r.boxes.xyxy.tolist()
        resultsList.append(image_id + box + conf)

df = pd.DataFrame(resultsList, columns = ["image_id", "x_min", "y_min", "x_max", "y_max", "conf"])
df.to_csv(saveDir, sep = ",", index = None)