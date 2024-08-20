# Mini_VinDr_CXR_Dataset
A subset of VinDr-CXR Dataset to facilitate preliminary experimentation and testing of code before using it on the main dataset.

Datasets:
1024_brightnessEQ_dataset - a dataset of images resized to 1024x1024 with padding, and brightness equalisation applied to it
original_dataset - the small original subset in dicom
dataset - the small original dataset converted to .png -> had issue with some of the images with color inverted to white background, black image
new_dataset - the small original dataset converted to .png -> issue with colour inversion fixed

Folders:
FULL_1024_PAD_annotations - a folder of annotations:
    - anno_test.csv and anno_train.csv should NOT be used as they contain invalid BB coordinates (less than 0, more than image dimensions)
    - 1024_PAD_allAnnoFIXED.csv are the above csv files combined with fixed BB coordinates
    - dimensions_test/train.csv are dimensions of the original images from the full datasets

YOLOv8 - folder of code written for YOLOv8 testing:
    - brightnessEQFIXED_best.pt - YOLOv8 pt file from training, can be used for inference
    - datasetToYOLOv8Format.py - converts a dataset for use with YOLOv8 - do not use the image conversion code, mostly just for BB conversions
    - model.yaml - file that is fed into YOLO with information about the dataset
    - transformImagesAndBB.py - depreciated for datasetToYOLOv8Format.py
    - detectionSpecificClasses.py - testing of YOLOv8 where we only detect for the classes that we know are already present in the image

Files:
EDA.ipynb - EDA on the original VinDr dataset
fixedOriginalBB.csv - merged test and train annotations fixed so no BB coordinates are out of range
fixOriginalBB.py - python code to do the above
imageModifications.py - python code to convert dataset to PNG, with padding, brightnessEQ, also contains code to grab original dimensions of dicom, create converted BB csv accounting for modifications
mergeBB.py - python code to merge bounding boxes that are overlapped/similar
viewer.py - image viewer that plots the image and bounding box
1024_PAD_allAnnoFIXED.csv - annotations for BB with 1024, pad, FIXED version that comes from fixed original annotations (no observations out of range)
mergedIOUAverage.csv - testing merging BB
mergedIOUEncompass.csv - testing merging BB