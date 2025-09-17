import glob
from pathlib import Path
import json
import pydicom


def get_dicom_metadata(images_dir: str):
    images = glob.glob(images_dir)
    metadata_dict = {}
    for image in images:
        image_id = Path(image).stem
        dicom = pydicom.dcmread(image, stop_before_pixels=True)
        metadata = dicom.to_json_dict()
        metadata_dict[image_id] = metadata
    return metadata_dict

if __name__ == "__main__":
    images_dir = ""
    outpath = "./train_metadata.json"
    metadata = get_dicom_metadata(images_dir=images_dir)
    json.dump(metadata, outpath)