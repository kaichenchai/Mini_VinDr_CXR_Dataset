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


def copy_images_timm_format(input_dir:str, output_base_path:str, df: pd.DataFrame, ext: str):
    normal_dir = os.path.join(output_base_path, "normal")
    cardiomegaly_dir = os.path.join(output_base_path, "cardiomegaly")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(cardiomegaly_dir, exist_ok=True)

    # Filter image IDs
    normal = df[df["cardiomegaly_flag"] == 0]["image_id"].tolist()
    cardiomegaly = df[df["cardiomegaly_flag"] == 1]["image_id"].tolist()

    # Move normal images
    for image_id in normal:
        src = os.path.join(input_dir, f"{image_id}{ext}")
        dst = os.path.join(normal_dir, f"{image_id}{ext}")
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} not found.")

    # Move cardiomegaly images
    for image_id in cardiomegaly:
        src = os.path.join(input_dir, f"{image_id}{ext}")
        dst = os.path.join(cardiomegaly_dir, f"{image_id}{ext}")
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} not found.")

if __name__ == "__main__":
    files = pd.read_csv("../explainability/cardiomegaly/generate_subset/cardiomegaly_subset_to_annotate.csv")
    copy_images_timm_format("/home/kai/mnt/VinDr_Subsets/cardiomegaly_subset/1024_padding_CLAHE/train",
                            "/home/kai/mnt/VinDr_Subsets/cardiomegaly_subset/timm_format/8_bit_png/train",
                            files,
                            ".png")
