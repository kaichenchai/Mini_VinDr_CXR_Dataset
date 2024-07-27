#performs modification to image intensity and enhances chest x-ray images

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image, ImageOps
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

#from https://www.kaggle.com/code/raddar/convert-dicom-to-np-array-the-correct-way/notebook
def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    
    #this seems to normalise data between 0-255 and converts it to int8/8bit, can then be saved as PNG    
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

#from https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
def resizeWithPadding(image, newDim):
    #image: np.array
    #newDim: tuple of (newx, newy) dimensions
    
    oldDim = (image.shape[0], image.shape[1]) #gets the dimensions of original img (y, x)
    print(oldDim)
    ratio = float(max(newDim)/max(oldDim)) #finds the tightest ratio
    print(ratio)
    convertedDim = tuple([int(x*ratio) for x in oldDim]) #makes a tuple of dim
    print(convertedDim)
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
    img = read_xray('original_dataset/train_subset/01a3c3d994d85ce5634d2d13c03fd4b0.dicom')
    print(img.shape)
    imgPadding = resizeWithPadding(img, (640,640))
    img = cv2.resize(img, (640, 640))

    #norm_img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    cv2.imshow("original", img)
    cv2.imshow("padding", imgPadding)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
