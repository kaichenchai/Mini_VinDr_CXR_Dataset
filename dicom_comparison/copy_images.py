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


def copy_images_timm_format(input_dir:str,
                            output_base_path:str,
                            df: pd.DataFrame,
                            col_flag: str,
                            ext: str):
    normal_dir = os.path.join(output_base_path, "normal")
    condition_dir = os.path.join(output_base_path, "condition")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(condition_dir, exist_ok=True)

    # Filter image IDs
    normal = df[df[col_flag] == 0]["image_id"].tolist()
    condition = df[df[col_flag] == 1]["image_id"].tolist()

    # Move normal images
    for image_id in normal:
        src = os.path.join(input_dir, f"{image_id}{ext}")
        dst = os.path.join(normal_dir, f"{image_id}{ext}")
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} not found.")

    # Move cardiomegaly images
    for image_id in condition:
        src = os.path.join(input_dir, f"{image_id}{ext}")
        dst = os.path.join(condition_dir, f"{image_id}{ext}")
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} not found.")

if __name__ == "__main__":
    train = pd.read_csv("./vindr_dataset_pneumothorax/vindr_pneumothorax_train.csv")
    val = pd.read_csv("./vindr_dataset_pneumothorax/vindr_pneumothorax_val.csv")

    """
    copy_images(source_dir = "/mnt/data/datasets/vindr-cxr/1.0.0/train",
                output_dir = "/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/vindr_dicom/train",
                filenames = train["image_id"].to_list(),
                ext = ".dicom" ) 
    copy_images(source_dir = "/mnt/data/datasets/vindr-cxr/1.0.0/train",
                output_dir = "/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/vindr_dicom/val",
                filenames = val["image_id"].to_list(),
                ext = ".dicom" )
    """

    copy_images_timm_format("/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/8_bit_png_norm/val",
                            "/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/timm_format/8_bit_png_norm/val",
                            val,
                            "pneumothorax_flag",
                            ".png")
    copy_images_timm_format("/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/8_bit_png_norm/train",
                            "/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/timm_format/8_bit_png_norm/train",
                            train,
                            "pneumothorax_flag",
                            ".png")

    copy_images_timm_format("/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/16_bit_png_norm/val",
                            "/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/timm_format/16_bit_png_norm/val",
                            val,
                            "pneumothorax_flag",
                            ".png")
    copy_images_timm_format("/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/16_bit_png_norm/train",
                            "/home/kai/mnt/VinDr_Subsets/pneumothorax_subsets/timm_format/16_bit_png_norm/train",
                            train,
                            "pneumothorax_flag",
                            ".png")
