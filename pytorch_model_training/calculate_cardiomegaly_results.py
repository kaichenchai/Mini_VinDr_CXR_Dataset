import argparse
import json
from PIL import Image
import os

import torchvision.transforms.functional

from train import model_loader, generate_transformations

from matplotlib import patches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_names = [fname for fname in os.listdir(image_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in self.image_names
        ]
        self.image_ids = [fname.split(".")[0]
                          for fname in self.image_names]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_id = self.image_ids[idx]
        image = Image.open(img_path).convert("L")  # Still return a PIL image
        if self.transform:
            image = self.transform(image)
        return image, image_id

def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for box, lab, score in zip(target['boxes'], target["labels"], target["scores"]):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        colour_map = {1: "r",
                      2: "b"}
        if score.item() > 0.5:
            rect = patches.Rectangle((x, y),
                                    width, height,
                                    linewidth = 2,
                                    edgecolor = colour_map[lab.item()],
                                    facecolor = 'none')

            # Draw the bounding box on top of the image
            a.add_patch(rect)
    plt.show()
    
def torch_to_pil(img):
    image = torchvision.transforms.functional.to_pil_image(pic=img)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, required=True, help="Weights to load to model")
    parser.add_argument("--images_path", type=str, required=True, help="Folder of images to parse")
    parser.add_argument("--output_path", type=str, required=True, help="Json output")
    parser.add_argument("--iou threshold", type=float, nargs="?", default=0.3, help="IOU threshold for NMS")
    args = parser.parse_args()
    trained_weights = torch.load(args.weights_path, map_location=torch.device('cpu'))
    model = model_loader(feature_extracting=False,
                         num_classes=3,
                         greyscale_single_channel=True,
                         imgsize=(1024,1024),
                         weights=trained_weights)
    transforms = generate_transformations(no_img_channels=1, has_bounding_boxes=False)
    dataset = ImageDataset(image_dir=args.images_path,
                            transform=transforms)
    dataset_loader = DataLoader(dataset=dataset, batch_size=1,
                                collate_fn=lambda x: tuple(zip(*x)))
    image_names = dataset.image_ids
    print(image_names)
    model.eval()
    with torch.no_grad():
        for data, image_id in dataset_loader:
            prediction = model(data)
            for img, pred in zip(data, prediction):
                plot_img_bbox(torch_to_pil(img), pred)
    
    with open(file=args.output_path, mode="w") as outfile:
        json.dumps(prediction, outfile)