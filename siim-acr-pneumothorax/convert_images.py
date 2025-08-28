import cv2
import os
import argparse


def copy_over_images(root_path: str, output_path: str, transformations:list=None):
    if transformations:
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    for root, dirs, files in os.walk(root_path):
        relative_path = os.path.relpath(root, root_path)
        destination = os.path.join(output_path, relative_path)
        os.makedirs(destination, exist_ok=True)
        for file in files:
            full_image_path = os.path.join(root, file)
            file_extension = os.path.splitext(full_image_path)[1].lower()
            if file_extension in (".png", ".jpeg", ".jpg"):
                image = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
                if transformations:
                    for transformation in transformations:
                        if transformation == "CLAHE":
                            image = clahe.apply(image)
                        if transformation == "histEQ":
                            image = cv2.equalizeHist(image)
                output_directory = os.path.join(destination, file)
                cv2.imwrite(output_directory, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--transformations", type=str, nargs='+')
    args = parser.parse_args()
    
    copy_over_images(args.root_path, args.output_path, transformations=args.transformations)

