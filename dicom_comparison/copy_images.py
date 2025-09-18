import copy
import os
import shutil
import pandas as pd


def copy_images(source_dir: str, output_dir: str, filenames: list, ext: str):
    """
    Copies image files from source_dir to output_dir based on a list of filenames and file extension.

    Args:
        source_dir (str): The directory where the original images are located.
        output_dir (str): The directory where the images will be copied to.
        filenames (list): A list of filenames (without extension) to copy.
        ext (str): The file extension (e.g., '.jpg', '.png').

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name in filenames:
        src_file = os.path.join(source_dir, name + ext)
        dst_file = os.path.join(output_dir, name + ext)

        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")
        else:
            print(f"File not found: {src_file}")


if __name__ == "__main__":
    files = pd.read_csv("../explainability/cardiomegaly/generate_subset/cardiomegaly_subset_to_annotate.csv")
    names = files["image_id"].to_list()
    copy_images("/mnt/data/datasets/vindr-cxr/1.0.0/train","/home/kai/mnt/VinDr_Subsets/cardiomegaly_subset/dicom",names,".dicom")