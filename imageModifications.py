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
def dicomToData(path, voi_lut = True, fix_monochrome = True):
    #dicom = pydicom.read_file(path) read_file has been repreciated
    dicom = pydicom.dcmread(path)
    
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
def resizeNoPadding(image, newDim):
    #image: np.array
    #newDim: tuple of (newx, newy) dimensions
    
    newImg = cv2.resize(image, newDim)
    
    return newImg

#from https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
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

#not doing this in one go, may try and revisit the idea eventually when cleaning up the code
def resizeWithPaddingTEST(image, newDim):
    #image: np.array
    #newDim: tuple of (newx, newy) dimensions
    #annotationDir: reads in the file of bounding box annotations and then 
    
    oldDim = (image.shape[0], image.shape[1]) #gets the dimensions of original img (y, x)
    ratio = float(max(newDim)/max(oldDim)) #finds the tightest ratio
    convertedDim = tuple([int(x*ratio) for x in oldDim]) #makes a tuple of dim
    newImg = cv2.resize(image, (convertedDim[1], convertedDim[0])) #converts image to size, need to swap to (x, y)
    

    color = [0, 0, 0] #fills with black
    newImg = cv2.copyMakeBorder(newImg, 200, 50, 100, 50, cv2.BORDER_CONSTANT,
        value=color)
    
    return newImg

#conversion to PNG from a directory of dicom files
#does not apply filters
def dirToPNG(inputDir, outputDir, resolution, equalise = False, CLAHE = False, padding = True):
    #check if the outputDir exists, if not make it
    os.makedirs(outputDir, exist_ok=True)
    filenames = os.listdir(inputDir)    
    for filename in filenames:
        if filename.endswith(".dicom"):
            fullpath = os.path.join(inputDir, filename)
            #converts images to np array
            data = dicomToData(fullpath)
            #equalising brightness histogram make sure to do first as otherwise padding affects the equalisation
            if equalise == True:
                data = cv2.equalizeHist(data)
            #uses CLAHE to perform histogram equalisation, better way to do as does it with a moving kernel
            if CLAHE == True:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                data = clahe.apply(data)
            #resizes to uniform size and adding padding
            if padding == True:
                data = resizeWithPadding(data, resolution)
            else:
                data = resizeNoPadding(data, resolution)
            #convert to image and then saves as png
            image = Image.fromarray(data, mode = "L")
            image.save(os.path.join(outputDir, filename.replace('.dicom', '.png')))
            print(f"{filename} has been converted with settings: equalise: {equalise}, CLAHE: {CLAHE}, padding: {padding}")
    print("All images have been converted")

#gets the dimensions of all the dicom files in a directory
#then converts annotations file to account for buffer
#the process can definitely be optimised to say the least
#NOTE if not all of the images from the CSV file are within the input folder then the CSV will output with some of the original annos still
def convertAnnotations(inputDir, annotationsDir = None, newDim = (1024, 1024), csvName = "annotations.csv"): #directory of files and desired resolution
    try:
        annotations = pd.read_csv(annotationsDir)
    except FileNotFoundError:
        print("File not found, try again")
    else:
        filenames = os.listdir(inputDir)    
        for filename in filenames:
            if filename.endswith(".dicom"):
                fullpath = os.path.join(inputDir, filename)
                dicom = pydicom.dcmread(fullpath)
    
                oldDim = (dicom[0x28, 0x10].value, dicom[0x28, 0x11].value) #gets the dimensions of original img (y, x)
                #print(oldDim)
                ratio = float(max(newDim)/max(oldDim)) #finds the tightest ratio
                #print(ratio)
                convertedDim = tuple([int(x*ratio) for x in oldDim]) #makes a tuple of dim
                #print(convertedDim)
               
                delta_w = newDim[1] - convertedDim[1] #as newDim and convertedDim in (y, x) format
                delta_h = newDim[0] - convertedDim[0]
                top, bottom = delta_h//2, delta_h-(delta_h//2)
                left, right = delta_w//2, delta_w-(delta_w//2)
                
                #print(left, right, top, bottom)
                
                #getting imageid from the original filename, slicing off the .dicom bit
                imageID = filename[:-6]

                #scaling bounding box coordinates
                #need to convert original bb coordinates to equivalent in smaller version
                annotations.loc[annotations["image_id"] == imageID, ("x_min", "y_min", "x_max", "y_max")] = annotations.loc[annotations["image_id"] == imageID, ("x_min", "y_min", "x_max", "y_max")]*ratio
                                            
                #modifying coordinates by shifting them left and up if there is a need to (for padding)
                #seems that NaN values are not affected by operations such as + - * /
                #padding is added from the top and from the left, so need to make appropriate changes for that
                annotations.loc[annotations["image_id"] == imageID, ("x_min","x_max")] = annotations.loc[annotations["image_id"] == imageID, ("x_min","x_max")]+left
                annotations.loc[annotations["image_id"] == imageID, ("y_min","y_max")] = annotations.loc[annotations["image_id"] == imageID, ("y_min","y_max")]+top
                
        annotations.to_csv(csvName, sep = ",", header=True, index = False)

#grab the original dimensions of a directory of dicom files and puts it in a csv
def getDimensions(inputDir, csvName = "dimensions.csv"):
    imageIDs = []
    xDims = []
    yDims = []
    try:
        filenames = os.listdir(inputDir)
    except FileNotFoundError:
        print("File not found, try again")
    else:    
        for filename in filenames:
            if filename.endswith(".dicom"):
                fullpath = os.path.join(inputDir, filename)
                dicom = pydicom.dcmread(fullpath)
                imageIDs.append(filename[:-6])
                xDims.append(dicom[0x28, 0x11].value) #number of columns (x)
                yDims.append(dicom[0x28, 0x10].value) #number of rows (y)
        dimensions = pd.DataFrame(zip(imageIDs, xDims, yDims), columns = ("image_id", "x_dim", "y_dim"))
        dimensions.to_csv(csvName, sep = ",", header = True, index = False)

#https://blog.roboflow.com/how-to-draw-a-bounding-box-label-python/
#https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
#for cv2 rectangle, (top-left), (bottom-right)
def drawBoundingBox(data, annotations):
    pass

#converting annotations from a CSV that contains image name, x_min/max, y_min/max and x/ydim as columns
#if we feed in dimensionsDir, then we assume that we don't have x/ydim in the annotations CSV
def convertAnnotationsFromCSV(annotationsDir = None, dimensionsDir = None, newDim = (1024, 1024), csvName = "annotations.csv"):
    try:
        annotations = pd.read_csv(annotationsDir)
        if dimensionsDir is not None:
            dimensions = pd.read_csv(dimensionsDir)
            annotations = pd.merge(annotations, dimensions, on = ["image_id", "image_id"])
    except FileNotFoundError as e:
        print(FileNotFoundError)
    else:
        annotations = annotations.dropna(axis = 0)
        groups = annotations.groupby(["image_id"], group_keys = False, as_index = False, sort = False)
        groupsList = []
        for group in groups:
            oldDimX = group[1].x_dim.values[0] #gets x dim for each group
            oldDimY = group[1].y_dim.values[0] #gets y dim for each group
            oldDim = (oldDimX, oldDimY)
            ratio = float(max(newDim)/max(oldDim))
            convertedDim = tuple([int(x*ratio) for x in oldDim])
            delta_w = newDim[0] - convertedDim[0]
            delta_h = newDim[1] - convertedDim[1]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            group[1].x_min = group[1].x_min*ratio + left
            group[1].x_max = group[1].x_max*ratio + left
            group[1].y_min = group[1].y_min*ratio + top
            group[1].y_max = group[1].y_max*ratio + top
            groupsList.append(group[1])
        output = pd.concat(groupsList, ignore_index=True)
        output.to_csv(csvName, index = None)
        
if __name__ == "__main__":
    
    """img = dicomToData('original_dataset/train_subset/01a3c3d994d85ce5634d2d13c03fd4b0.dicom')
    print(img.shape)
    imgPadding = resizeWithPadding(img, (640,640))
    #this method stretches, doesn't pad
    #img = cv2.resize(img, (640, 640))

    #brightness normalisation, but has already been done
    #norm_img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    
    #https://www.geeksforgeeks.org/histograms-equalization-opencv/?ref=lbp
    #brightness equalisation, helps to bring out some more detail
    imgEq = cv2.equalizeHist(imgPadding)
    
    #https://www.geeksforgeeks.org/python-image-blurring-using-opencv/
    #https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    imgGaussian = cv2.GaussianBlur(imgEq, (3,3), sigmaX=0)
    imgMedianBlurring = cv2.medianBlur(imgEq, 3)
    #generally maintains edges better than other blurring algorithms
    imgBilateralFiltering = cv2.bilateralFilter(imgEq, d = 7, sigmaColor=20, sigmaSpace=20)
    
    res = np.hstack((imgPadding, imgEq))
    
    cv2.imshow("comparison", res)
    cv2.imshow("gaussian", imgGaussian)
    cv2.imshow("median blurring", imgMedianBlurring)
    cv2.imshow("bilateral filtering", imgBilateralFiltering)

    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    #testing out clahe - we pretty much always want to use this
    dirToPNG("original_dataset/test_subset/","1024_CLAHE_pad/images/val_mode_L/", (1024, 1024), CLAHE=True, padding = True)

    #dirToPNG("original_dataset/test_subset/","1024_brightnessEQ_dataset/images/val/", (1024, 1024), equalise=True, padding = True)
    
    #dirToPNG("original_dataset/test_subset/","1024_original_noclahe_nopad/images/val/", (1024, 1024), CLAHE=False, padding = False)
    
    #convertAnnotations("original_dataset/train_subset/","original_dataset/annotations/annotations_train.csv", (1024, 1024), "annotationsTrain.csv")
    #convertAnnotations("original_dataset/test_subset/","original_dataset/annotations/annotations_test.csv", (1024, 1024), "annotationsTest.csv")
    
    #getDimensions("original_dataset/test_subset/", "dimensionsTest.csv")
    
    #convertAnnotationsFromCSV("fixedBBtrain.csv", newDim = (1024, 1024), csvName = "anno_train.csv")
    
    #convertAnnotationsFromCSV("fixedBBtest.csv", newDim = (1024, 1024), csvName = "anno_test.csv")
    
    pass