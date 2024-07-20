import numpy as np
import pydicom
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

imgDir = "C:/Users/engli/Documents/Y2S1/NSPC2001/Mini_VinDr_CXR_Dataset/new_dataset/images/train/"
labelDir = "C:/Users/engli/Documents/Y2S1/NSPC2001/Mini_VinDr_CXR_Dataset/new_dataset/labels/train/"

print(imgDir)
print(labelDir)

choice = None

while choice != "x":
    choice = input("Image filename (x to close): ")
    match choice:
        case "x":
            print("Closing")
        case _:
            try:
                dir = imgDir + choice
                print(dir)
                image = Image.open(dir)
                fig, ax = plt.subplots()
                print(image.size)
                ax.imshow(image)
                plt.show()
            except FileNotFoundError:
                print("Image not found in folder")