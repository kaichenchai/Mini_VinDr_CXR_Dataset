import glob
import os
from pathlib import Path
import numpy as np
import cv2
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pydicom

def dicom_to_array(path, voi_lut = True, bit_depth: int = 16):
    #dicom = pydicom.read_file(path) read_file has been repreciated
    dicom = pydicom.dcmread(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    
    data = data.astype(np.uint32)
    data = (data * 2**bit_depth) / 2**dicom.BitsStored
    if bit_depth == 16:
        data = data.astype(np.uint16)
    elif bit_depth == 8:
        data = data.astype(np.uint8)
    elif bit_depth == 4:
        data = data.astype(np.uint4)
    else:
        data = data.astype(np.uint8)

    return data

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

def dicom_to_png(images_dir: str,
                output_dir: str,
                resolution = (1024, 1024),
                bit_depth = 16,
                equalise = False,
                CLAHE = True,
                padding = True):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    filetypes = [".dcm", ".dicom"]
    images = []
    for filetype in filetypes:
        images.extend(glob.glob(f"{images_dir}/*{filetype}"))
    print(images)
    for image in images:
        image_id = Path(image).stem
        array = dicom_to_array(image, bit_depth = bit_depth)
        if equalise:
            array = cv2.equalizeHist(array)
        #uses CLAHE to perform histogram equalisation, better way to do as does it with a moving kernel
        if CLAHE:
            array = clahe.apply(array)
        #resizes to uniform size and adding padding
        array = resizeWithPadding(array, resolution)
        #convert to image and then saves as png
        image = cv2.imwrite(os.path.join(output_dir, f"{image_id}.png"), array, )
        

if __name__ == "__main__":
    dicom_to_png("/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/dicom/val/",
                 "/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/8_bit_png_norm/val", bit_depth = 8)
    dicom_to_png("/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/dicom/train/",
                 "/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/8_bit_png_norm/train", bit_depth = 8)
    dicom_to_png("/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/dicom/val/",
                 "/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/16_bit_png_norm/val", bit_depth = 16)
    dicom_to_png("/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/dicom/train/",
                 "/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/16_bit_png_norm/train", bit_depth = 16)
