from ultralytics import YOLO

modelDir = ""
sourceDir = ""
saveDir = ""

model = YOLO(modelDir)
results = model.predict(
    source = sourceDir,
    imgsz = 1024,
    conf = 0.0001,
    iou = 0.6,
    max_det = 1,
)

print(results)