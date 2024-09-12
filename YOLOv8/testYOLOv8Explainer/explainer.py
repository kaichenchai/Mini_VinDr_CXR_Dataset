from YOLOv8_Explainer import yolov8_heatmap, display_images
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os

model = yolov8_heatmap(
    weight="best.pt", 
        conf_threshold=0.4, 
        device = "cpu", 
        method = "EigenCAM", 
        layer=[10, 12, 14, 16, 18, -3],
        backward_type="all",
        ratio=0.02,
        show_box=True,
        renormalize=False,
)

imagelist = model(
    img_path="/location/image.jpg", 
    )

display_images(imagelist)