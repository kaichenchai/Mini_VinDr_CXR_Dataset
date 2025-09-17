import glob
from pathlib import Path
import json
from tqdm import tqdm
import pydicom

def get_dicom_metadata(images_dir: str):
    filetypes = [".dcm", ".dicom"]
    images = []
    for filetype in filetypes:
        images.extend(glob.glob(f"{images_dir}/*{filetype}"))
    metadata_dict = {}
    for image in tqdm(images):
        image_id = Path(image).stem
        dicom = pydicom.dcmread(image, stop_before_pixels=True)
        metadata = dicom.to_json_dict()
        metadata_dict[image_id] = metadata
    return metadata_dict

if __name__ == "__main__":
    images_dir = "/home/kai/mnt/siim-acr-pneumothorax/siim_acr_dataset/test/dicom_files/"
    outpath = "./siim_acr_test_metadata.json"
    metadata = get_dicom_metadata(images_dir=images_dir)
    with open(outpath, "w") as f:
        json.dump(metadata, f, indent=2)
