import os
import shutil
from typing import Dict
import pandas as pd


def json_annos_to_txt_files(input_output_dict: Dict[str, str]):
    """_summary_

    Args:
        input_output_dict (Dict[str, str]): mapping input .json of annos to .txt files in output

    Returns:
        bool: 
    """
    for input, output_path in input_output_dict.items():
        input_df = pd.read_json(input)
        for index, row in input_df.iterrows():
            file_name = f"{row["image_id"]}.txt"
            bb_string = row["box"]
            if bb_string:
                txtFilePath = os.path.join(output_path, file_name)
                with open(txtFilePath, "a") as file:
                    file.write(f"{row["box"]}\n")
    return True

def move_images(source_dir: str,
                json_ids_to_output_dir: Dict[str, str],
                image_id_key_in_json: str = "image_id",
                image_ext: str = ".png"):
    if os.path.isdir(source_dir):
        for json_with_ids, output_dir in json_ids_to_output_dir.items():
            df = pd.read_json(json_with_ids)
            ids_to_move = df[image_id_key_in_json].to_list()
            for id in ids_to_move:
                image_filename = id + image_ext
                original_path = os.path.join(source_dir, image_filename)
                new_path = os.path.join(output_dir, image_filename)
                shutil.copyfile(original_path, new_path)
    else:
        raise ValueError(f"Source dir {source_dir} is not a path")
    return True

if __name__ == "__main__":
    input_files_with_outputs = {"./obb_annotations_stratified_split/annos_with_areas/25_percentile/yolo_format/train_yolo_format.json": "/home/kai/mnt/siim-acr-pneumothorax/yolo_format/siim_acr_512_yolo/labels_25_percentile/train",
                                "./obb_annotations_stratified_split/annos_with_areas/25_percentile/yolo_format/val_yolo_format.json": "/home/kai/mnt/siim-acr-pneumothorax/yolo_format/siim_acr_512_yolo/labels_25_percentile/val",
                                "./obb_annotations_stratified_split/annos_with_areas/25_percentile/yolo_format/test_yolo_format.json": "/home/kai/mnt/siim-acr-pneumothorax/yolo_format/siim_acr_512_yolo/labels_25_percentile/test"}
    print(json_annos_to_txt_files(input_files_with_outputs))
