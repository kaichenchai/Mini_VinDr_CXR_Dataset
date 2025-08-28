import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class PneumothoraxDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 annotations_json_path: str,
                 transform = None,
                 images_size: tuple = None):
               
        self.images_dir = images_dir
        self.annotations_json_path = annotations_json_path
        self.transforms = transform
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith(".png")]
        self.image_files.sort()
        self.images_size = images_size
        self.annotations = pd.read_json(self.annotations_json_path)
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, index):
            pass